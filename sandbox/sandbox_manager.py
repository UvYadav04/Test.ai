import hashlib
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass

import docker
from docker.errors import ImageNotFound, NotFound

from sandbox.path_resolver import InvalidArtifactIdError, validate_segment
from sandbox.sandbox_client import SandboxClient, SandboxClientError

logger = logging.getLogger("sandbox.manager")

_SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))

_SANDBOX_SOURCE_FILES = ("Dockerfile", "sandbox_server.py", "execution_engine.py", "path_resolver.py")


def _sandbox_image_tag() -> str:
    h = hashlib.sha256()
    for name in _SANDBOX_SOURCE_FILES:
        with open(os.path.join(_SANDBOX_DIR, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


IMAGE_NAME = f"dataanalyzer-sandbox:{_sandbox_image_tag()}"

PARQUET_VOLUME_NAME = os.environ.get("PARQUET_VOLUME_NAME", "dataanalyzer_parquet_data")

SANDBOX_SOCKET_VOLUME_NAME = os.environ.get("SANDBOX_SOCKET_VOLUME_NAME", "dataanalyzer_sandbox_sockets")
SANDBOX_SOCKET_CONTAINER_MOUNT = os.environ.get("SANDBOX_SOCKET_ROOT", "/data/sandbox_sockets")

DEFAULT_MIN_POOL_SIZE = int(os.environ.get("SANDBOX_POOL_MIN_SIZE", "2"))
DEFAULT_MAX_POOL_SIZE = int(os.environ.get("SANDBOX_POOL_MAX_SIZE", "8"))
DEFAULT_IDLE_TIMEOUT_SECONDS = int(os.environ.get("SANDBOX_IDLE_TIMEOUT_SECONDS", "300"))
DEFAULT_HEALTH_TIMEOUT_SECONDS = float(os.environ.get("SANDBOX_HEALTH_TIMEOUT_SECONDS", "15"))
DEFAULT_REAP_INTERVAL_SECONDS = int(os.environ.get("SANDBOX_REAP_INTERVAL_SECONDS", "30"))
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = float(os.environ.get("SANDBOX_ACQUIRE_TIMEOUT_SECONDS", "20"))

try:
    from prometheus_client import REGISTRY, Counter, Gauge, Histogram

    def _metric(cls, name, documentation, **kwargs):
        try:
            return cls(name, documentation, **kwargs)
        except ValueError:
            existing = REGISTRY._names_to_collectors.get(name)
            if existing is not None:
                return existing
            raise

    _POOL_IDLE = _metric(Gauge, "sandbox_pool_idle", "Idle sandboxes currently in the pool")
    _POOL_BUSY = _metric(Gauge, "sandbox_pool_busy", "Busy sandboxes currently in the pool")
    _POOL_TOTAL = _metric(
        Gauge, "sandbox_pool_total", "Total sandboxes currently in the pool (idle+busy+pending)"
    )
    _SCALE_UP = _metric(
        Counter, "sandbox_pool_scale_up_total",
        "Sandbox pool scale-up events (new container beyond min_size)",
    )
    _DESTROYED_IDLE = _metric(
        Counter, "sandbox_pool_destroyed_idle_total", "Idle sandboxes destroyed by the idle reaper"
    )
    _FAILED_EXEC = _metric(Counter, "sandbox_pool_failed_executions_total", "Executions that failed")
    _UNHEALTHY_REPLACED = _metric(
        Counter, "sandbox_pool_unhealthy_replacements_total", "Unhealthy sandboxes evicted and replaced"
    )
    _WAIT_HIST = _metric(Histogram, "sandbox_pool_wait_seconds", "Time a caller waited to acquire a sandbox")
    _EXEC_HIST = _metric(Histogram, "sandbox_pool_execution_seconds", "Sandbox code-execution time")
    _STARTUP_HIST = _metric(
        Histogram, "sandbox_pool_startup_seconds", "Sandbox container creation+health-check time"
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not installed - sandbox pool metrics will only be available via get_metrics()")


class SandboxManagerError(RuntimeError):
    pass


class SandboxPoolExhausted(SandboxManagerError):
    """Raised when no idle sandbox is available, the pool is already at max_size, and no
    sandbox became free before the acquire timeout elapsed."""


IDLE = "idle"
BUSY = "busy"


class _SandboxHandle:
    __slots__ = (
        "sandbox_id", "container", "client", "socket_path",
        "created_at", "last_used_at", "execution_count", "state",
    )

    def __init__(self, sandbox_id: str, container, client: SandboxClient, socket_path: str):
        self.sandbox_id = sandbox_id
        self.container = container
        self.client = client
        self.socket_path = socket_path
        self.created_at = time.time()
        self.last_used_at = self.created_at
        self.execution_count = 0
        self.state = BUSY  # handles are only ever handed out already-busy; see _create_sandbox callers

    @property
    def uptime_s(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_s(self) -> float:
        return time.time() - self.last_used_at

    def snapshot(self) -> dict:
        return {
            "sandbox_id": self.sandbox_id,
            "state": self.state,
            "uptime_s": round(self.uptime_s, 1),
            "idle_s": round(self.idle_s, 1),
            "execution_count": self.execution_count,
            "container_id": getattr(self.container, "short_id", None),
        }


@dataclass
class _Metrics:
    scale_up_events: int = 0
    destroyed_idle: int = 0
    failed_executions: int = 0
    unhealthy_replacements: int = 0
    total_created: int = 0
    total_destroyed: int = 0
    wait_time_ms_sum: float = 0.0
    wait_time_count: int = 0
    exec_time_ms_sum: float = 0.0
    exec_time_count: int = 0
    startup_time_ms_sum: float = 0.0
    startup_time_count: int = 0

    def record_wait(self, ms: float) -> None:
        self.wait_time_ms_sum += ms
        self.wait_time_count += 1
        if _PROMETHEUS_AVAILABLE:
            _WAIT_HIST.observe(ms / 1000)

    def record_exec(self, ms: float) -> None:
        self.exec_time_ms_sum += ms
        self.exec_time_count += 1
        if _PROMETHEUS_AVAILABLE:
            _EXEC_HIST.observe(ms / 1000)

    def record_startup(self, ms: float) -> None:
        self.startup_time_ms_sum += ms
        self.startup_time_count += 1
        if _PROMETHEUS_AVAILABLE:
            _STARTUP_HIST.observe(ms / 1000)

    @staticmethod
    def _avg(total: float, count: int) -> float:
        return round(total / count, 1) if count else 0.0

    def snapshot(self) -> dict:
        return {
            "avg_wait_time_ms": self._avg(self.wait_time_ms_sum, self.wait_time_count),
            "avg_execution_time_ms": self._avg(self.exec_time_ms_sum, self.exec_time_count),
            "avg_startup_time_ms": self._avg(self.startup_time_ms_sum, self.startup_time_count),
            "scale_up_events": self.scale_up_events,
            "destroyed_idle_sandboxes": self.destroyed_idle,
            "failed_executions": self.failed_executions,
            "unhealthy_replacements": self.unhealthy_replacements,
            "total_created": self.total_created,
            "total_destroyed": self.total_destroyed,
        }


class SandboxManager:

    MAX_EXECUTE_ATTEMPTS = 2

    def __init__(
        self,
        socket_root: str,
        image: str = IMAGE_NAME,
        mem_limit: str = "512m",
        nano_cpus: int = 1_000_000_000,
        min_size: int = DEFAULT_MIN_POOL_SIZE,
        max_size: int = DEFAULT_MAX_POOL_SIZE,
        idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS,
        health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
        reap_interval_seconds: int = DEFAULT_REAP_INTERVAL_SECONDS,
        acquire_timeout_seconds: float = DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
        start_reaper: bool = True,
    ):
        if min_size < 0:
            raise ValueError("min_size must be >= 0")
        if max_size < 1 or max_size < min_size:
            raise ValueError("max_size must be >= 1 and >= min_size")

        self.socket_root = os.path.abspath(socket_root)
        os.makedirs(self.socket_root, exist_ok=True)
        self.image = image
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout_seconds = idle_timeout_seconds
        self.health_timeout_seconds = health_timeout_seconds
        self.reap_interval_seconds = reap_interval_seconds
        self.acquire_timeout_seconds = acquire_timeout_seconds

        self._idle: dict[str, _SandboxHandle] = {}
        self._busy: dict[str, _SandboxHandle] = {}
        self._pending = 0  # containers currently mid-creation; counted against max_size
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._client = None
        self._metrics = _Metrics()

        self._reaper_stop = threading.Event()
        self._reaper_thread = None
        if start_reaper:
            self._reaper_thread = threading.Thread(
                target=self._reap_loop, name="sandbox-pool-reaper", daemon=True,
            )
            self._reaper_thread.start()

        logger.info(
            "SandboxManager pool started: image=%s min_size=%d max_size=%d idle_timeout=%ds "
            "reap_interval=%ds acquire_timeout=%.0fs socket_root=%s",
            image, min_size, max_size, idle_timeout_seconds, reap_interval_seconds,
            acquire_timeout_seconds, self.socket_root,
        )


    @property
    def client(self):
        if self._client is None:
            try:
                self._client = docker.from_env()
            except Exception as exc:
                raise SandboxManagerError(
                    "could not connect to Docker - is Docker Desktop/daemon running?"
                ) from exc
        return self._client

    def ensure_image(self) -> None:
        start = time.perf_counter()
        try:
            self.client.images.get(self.image)
            logger.info(
                "ensure_image: image %s already exists (checked in %.1fms) - reusing it",
                self.image, (time.perf_counter() - start) * 1000,
            )
            return
        except ImageNotFound:
            logger.info("ensure_image: image %s not found - building it now", self.image)

        try:
            logger.info("ensure_image: building image %s from build context %s", self.image, _SANDBOX_DIR)
            image, logs = self.client.images.build(path=_SANDBOX_DIR, tag=self.image, rm=True)
            for chunk in logs:
                if "stream" in chunk:
                    logger.info(chunk["stream"].rstrip())
                elif "error" in chunk:
                    logger.error(chunk["error"])
            logger.info(
                "ensure_image: image %s built successfully in %.2fs (id=%s)",
                self.image, time.perf_counter() - start, getattr(image, "short_id", "?"),
            )
        except Exception:
            logger.exception("ensure_image: failed to build sandbox image %s", self.image)
            raise


    def _total_locked(self) -> int:
        """Caller must hold self._lock."""
        return len(self._idle) + len(self._busy) + self._pending

    def _publish_gauges_locked(self) -> None:
        """Caller must hold self._lock."""
        if not _PROMETHEUS_AVAILABLE:
            return
        _POOL_IDLE.set(len(self._idle))
        _POOL_BUSY.set(len(self._busy))
        _POOL_TOTAL.set(self._total_locked())

    def warm_pool(self) -> None:
       
        self.ensure_image()
        with self._lock:
            deficit = self.min_size - self._total_locked()
            if deficit <= 0:
                logger.info("warm_pool: pool already has %d sandbox(es), min_size=%d - nothing to do",
                            self._total_locked(), self.min_size)
                return
            self._pending += deficit

        logger.info("warm_pool: creating %d sandbox(es) to reach min_size=%d", deficit, self.min_size)
        results = []
        results_lock = threading.Lock()

        def _spawn():
            try:
                handle = self._create_sandbox()
                with results_lock:
                    results.append(handle)
            except Exception:
                logger.exception("warm_pool: failed to create a warm sandbox")

        threads = [threading.Thread(target=_spawn, daemon=True) for _ in range(deficit)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with self._lock:
            self._pending -= deficit
            for handle in results:
                handle.state = IDLE
                self._idle[handle.sandbox_id] = handle
            self._publish_gauges_locked()
            self._not_empty.notify_all()

        logger.info("warm_pool: pool now has %d idle sandbox(es) (%d/%d requested succeeded)",
                    len(self._idle), len(results), deficit)


    def acquire(self, timeout: float = None) -> _SandboxHandle:
        timeout = self.acquire_timeout_seconds if timeout is None else timeout
        deadline = time.perf_counter() + timeout
        wait_start = time.perf_counter()

        with self._lock:
            while True:
                if self._idle:
                    sandbox_id, handle = self._idle.popitem()
                    handle.state = BUSY
                    self._busy[sandbox_id] = handle
                    self._publish_gauges_locked()
                    self._metrics.record_wait((time.perf_counter() - wait_start) * 1000)
                    return handle

                if self._total_locked() < self.max_size:
                    self._pending += 1
                    self._metrics.scale_up_events += 1
                    if _PROMETHEUS_AVAILABLE:
                        _SCALE_UP.inc()
                    self._publish_gauges_locked()
                    break  

                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise SandboxPoolExhausted(
                        f"no idle sandbox available and pool is at max_size={self.max_size} "
                        f"(waited {timeout:.1f}s)"
                    )
                self._not_empty.wait(timeout=remaining)

        logger.info("pool scale-up: creating a new sandbox beyond min_size=%d (max_size=%d)",
                    self.min_size, self.max_size)
        try:
            handle = self._create_sandbox()
        except Exception:
            with self._lock:
                self._pending -= 1
                self._publish_gauges_locked()
                self._not_empty.notify_all()
            raise

        with self._lock:
            self._pending -= 1
            handle.state = BUSY
            self._busy[handle.sandbox_id] = handle
            self._publish_gauges_locked()
        self._metrics.record_wait((time.perf_counter() - wait_start) * 1000)
        logger.info("pool scale-up: sandbox=%s ready (pool now %d idle / %d busy)",
                    handle.sandbox_id, len(self._idle), len(self._busy))
        return handle

    def release(self, handle: _SandboxHandle, healthy: bool = True) -> None:
        with self._lock:
            self._busy.pop(handle.sandbox_id, None)

        if healthy:
            try:
                reset_result = handle.client.reset()
                healthy = bool((reset_result or {}).get("healthy", True))
                if not healthy:
                    logger.warning(
                        "release: sandbox=%s reported an unclean reset (leaked thread/process?) "
                        "- discarding it instead of returning it to the pool", handle.sandbox_id,
                    )
            except SandboxClientError as exc:
                logger.warning("release: sandbox=%s failed to reset, discarding: %s", handle.sandbox_id, exc)
                healthy = False

        handle.last_used_at = time.time()

        if not healthy:
            self._metrics.unhealthy_replacements += 1
            if _PROMETHEUS_AVAILABLE:
                _UNHEALTHY_REPLACED.inc()
            self._destroy_handle(handle)
            with self._lock:
                self._publish_gauges_locked()
                self._not_empty.notify_all()
            self._maybe_replace_to_min()
            return

        handle.state = IDLE
        with self._lock:
            self._idle[handle.sandbox_id] = handle
            self._publish_gauges_locked()
            self._not_empty.notify_all()

    def _maybe_replace_to_min(self) -> None:
        with self._lock:
            deficit = self.min_size - self._total_locked()
            if deficit <= 0:
                return
            self._pending += 1
            self._publish_gauges_locked()
        try:
            handle = self._create_sandbox()
        except Exception:
            with self._lock:
                self._pending -= 1
                self._publish_gauges_locked()
            logger.exception("failed to create replacement sandbox after an unhealthy eviction")
            return
        handle.state = IDLE
        with self._lock:
            self._pending -= 1
            self._idle[handle.sandbox_id] = handle
            self._publish_gauges_locked()
            self._not_empty.notify_all()
        logger.info("replaced unhealthy sandbox: pool back at min_size=%d", self.min_size)

    
    def execute(
        self, code: str, tables: dict, workspace_id: str,
        timeout_seconds: float = None, tag: str = None,
    ) -> dict:
       
        last_error = None

        for attempt in range(1, self.MAX_EXECUTE_ATTEMPTS + 1):
            try:
                handle = self.acquire()
            except SandboxPoolExhausted as exc:
                last_error = exc
                logger.warning(
                    "execute: attempt %d/%d could not acquire a sandbox (tag=%s): %s",
                    attempt, self.MAX_EXECUTE_ATTEMPTS, tag, exc,
                )
                continue

            logger.info(
                "execute: attempt %d/%d sandbox=%s (tag=%s, workspace=%s, tables=%d, code_chars=%d)",
                attempt, self.MAX_EXECUTE_ATTEMPTS, handle.sandbox_id, tag, workspace_id, len(tables), len(code),
            )

            t0 = time.perf_counter()
            healthy = True
            try:
                result = handle.client.execute(code, tables, workspace_id, timeout_seconds=timeout_seconds)
                handle.execution_count += 1
                self._metrics.record_exec((time.perf_counter() - t0) * 1000)
                logger.info(
                    "execute: attempt %d/%d sandbox=%s completed in %.1fms",
                    attempt, self.MAX_EXECUTE_ATTEMPTS, handle.sandbox_id, (time.perf_counter() - t0) * 1000,
                )
                return result
            except SandboxClientError as exc:
                last_error = exc
                healthy = False
                self._metrics.failed_executions += 1
                if _PROMETHEUS_AVAILABLE:
                    _FAILED_EXEC.inc()
                logger.warning(
                    "execute: attempt %d/%d sandbox=%s failed after %.1fms (tag=%s): %s - discarding "
                    "this sandbox, next attempt (if any) gets a different one",
                    attempt, self.MAX_EXECUTE_ATTEMPTS, handle.sandbox_id,
                    (time.perf_counter() - t0) * 1000, tag, exc,
                )
                continue
            finally:
                self.release(handle, healthy=healthy)

        logger.error("execute: gave up after %d attempt(s) (tag=%s): %s", self.MAX_EXECUTE_ATTEMPTS, tag, last_error)
        return {"stdout": "", "saved": [], "error": str(last_error)}


    def _create_sandbox(self) -> _SandboxHandle:
        sandbox_id = uuid.uuid4().hex[:12]
        logger.info("sandbox creation: starting new pool container id=%s", sandbox_id)

        socket_filename = f"{sandbox_id}.sock"
        host_socket_path = os.path.join(self.socket_root, socket_filename)
        if os.path.exists(host_socket_path):
            logger.warning("sandbox creation: removing stale socket file %s", host_socket_path)
            try:
                os.remove(host_socket_path)
            except OSError:
                logger.exception("failed to remove stale socket file %s", host_socket_path)

        volumes = {
            PARQUET_VOLUME_NAME: {"bind": "/data/parquet", "mode": "rw"},
            SANDBOX_SOCKET_VOLUME_NAME: {"bind": SANDBOX_SOCKET_CONTAINER_MOUNT, "mode": "rw"},
        }
        environment = {
            "SANDBOX_ID": sandbox_id,
            "SANDBOX_SOCKET_ROOT": SANDBOX_SOCKET_CONTAINER_MOUNT,
        }

        create_start = time.perf_counter()
        container = self.client.containers.run(
            self.image,
            detach=True,
            network_disabled=True,
            mem_limit=self.mem_limit,
            nano_cpus=self.nano_cpus,
            volumes=volumes,
            environment=environment,
            labels={"dataanalyzer.pool": "true", "dataanalyzer.sandbox_id": sandbox_id},
        )
        container.reload()

        client = SandboxClient(host_socket_path)
        try:
            self._wait_for_health(client, container, sandbox_id)
        except Exception:
            logger.warning("sandbox creation: id=%s failed to become healthy - removing it", sandbox_id)
            try:
                # container.remove(force=True)
                container.stop()
            except Exception:
                pass
            raise

        create_s = time.perf_counter() - create_start
        self._metrics.record_startup(create_s * 1000)
        self._metrics.total_created += 1
        logger.info(
            "sandbox creation: id=%s container=%s ready in %.3fs",
            sandbox_id, container.short_id, create_s,
        )
        return _SandboxHandle(sandbox_id, container, client, host_socket_path)

    def _wait_for_health(self, client: SandboxClient, container, sandbox_id: str) -> None:
        deadline = time.perf_counter() + self.health_timeout_seconds
        attempt = 0
        last_error = None
        while time.perf_counter() < deadline:
            attempt += 1
            try:
                container.reload()
                if container.status != "running":
                    logs = container.logs(tail=50).decode("utf-8", errors="replace")
                    raise SandboxManagerError(
                        f"sandbox container {sandbox_id} exited before becoming healthy "
                        f"(status={container.status}): {logs}"
                    )
                client.health(timeout=1.5)
                logger.info(
                    "sandbox=%s container=%s UDS connection established after %d attempt(s)",
                    sandbox_id, container.short_id, attempt,
                )
                return
            except SandboxManagerError:
                raise
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)

        logger.error(
            "sandbox health check: id=%s container=%s gave up after %d attempt(s) over %.1fs: %s",
            sandbox_id, container.short_id, attempt, self.health_timeout_seconds, last_error,
        )
        raise SandboxManagerError(
            f"sandbox {sandbox_id} did not become healthy within "
            f"{self.health_timeout_seconds}s ({attempt} attempt(s)): {last_error}"
        )

    def _destroy_handle(self, handle: _SandboxHandle) -> None:
        try:
            handle.client.shutdown()
        except Exception:
            pass
        try:
            handle.container.remove(force=True)
            logger.info("sandbox cleanup: container %s removed (sandbox=%s)",
                        handle.container.short_id, handle.sandbox_id)
        except NotFound:
            pass
        except Exception:
            logger.exception("failed to remove sandbox container %s", handle.container.short_id)
        finally:
            handle.client.close()
        try:
            if os.path.exists(handle.socket_path):
                os.remove(handle.socket_path)
        except OSError:
            logger.exception("failed to remove socket file %s", handle.socket_path)
        self._metrics.total_destroyed += 1

    # -------------------------------------------------------------------------- reaper ----

    def _reap_loop(self) -> None:
        logger.info("pool reaper started (interval=%ds, idle_timeout=%ds, min_size=%d)",
                    self.reap_interval_seconds, self.idle_timeout_seconds, self.min_size)
        while not self._reaper_stop.wait(self.reap_interval_seconds):
            try:
                self._reap_once()
            except Exception:
                logger.exception("pool reaper: unexpected error during reap pass")

    def _reap_once(self) -> None:
        now = time.time()

        # 1) idle sandboxes that have overstayed get destroyed, but never below min_size.
        with self._lock:
            idle_oldest_first = sorted(self._idle.values(), key=lambda h: h.last_used_at)
            capacity_above_min = self._total_locked() - self.min_size
            to_reap = []
            for h in idle_oldest_first:
                if capacity_above_min <= 0:
                    break
                if now - h.last_used_at > self.idle_timeout_seconds:
                    to_reap.append(h)
                    capacity_above_min -= 1
            for h in to_reap:
                self._idle.pop(h.sandbox_id, None)
            self._publish_gauges_locked()

        for h in to_reap:
            logger.info(
                "idle reap: sandbox=%s idle for %.0fs (> %ds) - destroying (pool stays >= min_size=%d)",
                h.sandbox_id, now - h.last_used_at, self.idle_timeout_seconds, self.min_size,
            )
            self._metrics.destroyed_idle += 1
            if _PROMETHEUS_AVAILABLE:
                _DESTROYED_IDLE.inc()
            self._destroy_handle(h)

        # 2) health-check whatever idle sandboxes remain; evict+replace anything unresponsive.
        # Busy sandboxes are skipped - they're actively serving a request and execute()/release()
        # already treats a failed call as unhealthy.
        with self._lock:
            idle_snapshot = list(self._idle.values())
        unhealthy = [h for h in idle_snapshot if not self._check_health(h)]

        if unhealthy:
            with self._lock:
                for h in unhealthy:
                    self._idle.pop(h.sandbox_id, None)
                self._publish_gauges_locked()
            for h in unhealthy:
                logger.warning("health check: idle sandbox=%s is unhealthy - evicting", h.sandbox_id)
                self._metrics.unhealthy_replacements += 1
                if _PROMETHEUS_AVAILABLE:
                    _UNHEALTHY_REPLACED.inc()
                self._destroy_handle(h)
            self._maybe_replace_to_min()

        with self._lock:
            self._not_empty.notify_all()

    def _check_health(self, handle: _SandboxHandle) -> bool:
        try:
            handle.container.reload()
            if handle.container.status != "running":
                return False
            handle.client.health(timeout=2.0)
            return True
        except Exception as exc:
            logger.debug("health check failed for sandbox=%s: %s", handle.sandbox_id, exc)
            return False

    # -------------------------------------------------------------------------- metrics ----

    def get_metrics(self) -> dict:
        with self._lock:
            snapshot = {
                "total_sandboxes": self._total_locked(),
                "idle_sandboxes": len(self._idle),
                "busy_sandboxes": len(self._busy),
                "pending_sandboxes": self._pending,
                "min_size": self.min_size,
                "max_size": self.max_size,
            }
        snapshot.update(self._metrics.snapshot())
        return snapshot

    def get_pool_snapshot(self) -> list:
        with self._lock:
            return [h.snapshot() for h in list(self._idle.values()) + list(self._busy.values())]

    # -------------------------------------------------------------------------- shutdown ----

    def shutdown_all(self) -> None:
        self._reaper_stop.set()
        with self._lock:
            handles = list(self._idle.values()) + list(self._busy.values())
            self._idle.clear()
            self._busy.clear()
            self._publish_gauges_locked()
        logger.info("SandboxManager shutdown: releasing %d sandbox(es)", len(handles))
        for handle in handles:
            self._destroy_handle(handle)


_manager_lock = threading.Lock()
_default_manager: "SandboxManager | None" = None


def get_manager(socket_root: str = None, **kwargs) -> SandboxManager:
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                root = socket_root or os.environ.get("SANDBOX_SOCKET_ROOT") or os.path.join(
                    _SANDBOX_DIR, "..", "data", "sandbox_sockets",
                )
                _default_manager = SandboxManager(socket_root=root, **kwargs)
    return _default_manager
