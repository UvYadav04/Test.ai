"""Host-side lifecycle manager for persistent sandbox containers - one container per
investigation_id, reused across every run_python() call that investigation makes (see
sandbox_executor.py's PythonSandbox, the only intended caller of this module besides
worker_service's own startup/shutdown wiring).

Why keyed by investigation_id, not workspace_id or a global pool: an investigation is one
orchestrator run (worker_service/tasks/investigation.py's run_investigation) - it may delegate
to the Tabular Agent, and therefore call run_python, many times over its lifetime, each time
through a NEW TabularAgent/TabularTools instance (see agents/tabular/agent.py). Keying the cache
by investigation_id is what lets all of those short-lived instances land on the exact same warm
container instead of each paying Docker-create + interpreter-boot again. Keying it any more
broadly (e.g. by workspace_id, which usually outlives a single investigation) would mean two
investigations running concurrently in the same workspace could execute inside the SAME
container at the same time - that's the one thing this module must never allow (see the
class docstring below) - so investigation_id, which this codebase already treats as unique per
chat turn, is the right granularity.

docker-outside-of-docker note: same as sandbox_executor.py/path_resolver.py before this file
existed - `self.client` is `docker.from_env()`, which talks to the HOST's Docker daemon when
worker_service itself runs containerized with /var/run/docker.sock bind-mounted in. Both the
parquet volume AND the new sandbox-socket volume below are named Docker volumes (not bind
mounts) for exactly the reason those files already document: the host daemon resolves a volume
NAME identically regardless of which container asks for it, so worker_service and every sandbox
sibling container it creates can share one without agreeing on any host filesystem path.
"""
import logging
import os
import threading
import time

import docker
from docker.errors import ImageNotFound, NotFound

from sandbox.path_resolver import InvalidArtifactIdError, validate_segment
from sandbox.sandbox_client import SandboxClient, SandboxClientError

logger = logging.getLogger("sandbox.manager")

IMAGE_NAME = "dataanalyzer-sandbox:latest"
_SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))

# Same named-volume convention as sandbox_executor.py's PARQUET_VOLUME_NAME - override if your
# deployment's COMPOSE_PROJECT_NAME/directory name isn't "dataanalyzer" (`docker volume ls`
# shows the real runtime name).
PARQUET_VOLUME_NAME = os.environ.get("PARQUET_VOLUME_NAME", "dataanalyzer_parquet_data")

# Backs the .sock files this module creates/waits on and every sandbox container serves from -
# see docker-compose.yml's `sandbox_sockets` volume. Mounted at SANDBOX_SOCKET_CONTAINER_MOUNT
# inside every sandbox container; this process's OWN mount point for the same volume is
# `socket_root` (constructor arg, from worker_service's SANDBOX_SOCKET_ROOT - see
# engine_bootstrap.py), independent of this container-side path.
SANDBOX_SOCKET_VOLUME_NAME = os.environ.get("SANDBOX_SOCKET_VOLUME_NAME", "dataanalyzer_sandbox_sockets")
SANDBOX_SOCKET_CONTAINER_MOUNT = "/shared"

DEFAULT_IDLE_TIMEOUT_SECONDS = int(os.environ.get("SANDBOX_IDLE_TIMEOUT_SECONDS", "600"))
DEFAULT_HEALTH_TIMEOUT_SECONDS = float(os.environ.get("SANDBOX_HEALTH_TIMEOUT_SECONDS", "15"))
DEFAULT_REAP_INTERVAL_SECONDS = int(os.environ.get("SANDBOX_REAP_INTERVAL_SECONDS", "60"))


class SandboxManagerError(RuntimeError):
    pass


class _SandboxHandle:
    """One live container + its client, cached in SandboxManager._sandboxes. `lock` serializes
    concurrent execute() calls FOR THIS INVESTIGATION ONLY (arq's max_jobs can run this worker's
    jobs concurrently, and nothing stops two tool calls within one investigation's orchestrator
    loop from racing in unusual code paths) - the sandbox server itself handles one request at a
    time regardless, but taking this lock host-side gives cleaner timeout/error attribution than
    letting two requests queue invisibly inside Uvicorn."""

    __slots__ = ("investigation_id", "container", "client", "socket_path", "created_at", "last_used_at", "lock")

    def __init__(self, investigation_id: str, container, client: SandboxClient, socket_path: str):
        self.investigation_id = investigation_id
        self.container = container
        self.client = client
        self.socket_path = socket_path
        self.created_at = time.time()
        self.last_used_at = self.created_at
        self.lock = threading.Lock()


class SandboxManager:
    """Different investigations must NEVER share execution state: each investigation_id gets its
    own container, its own socket file, its own ExecutionEngine instance inside that container -
    never a container (or a warm namespace/DuckDB connection inside one) reused across two
    different investigation_ids. That isolation is what makes this safe to reuse a container
    WITHIN one investigation at all (see module docstring) - the boundary this class enforces is
    per-investigation, not per-call.

    Lifecycle: get_or_create() creates a container + waits for /health on the first call for a
    given investigation_id, and returns the same cached SandboxClient on every later call.
    release() destroys a specific investigation's sandbox immediately (called from
    worker_service/tasks/investigation.py when that investigation ends). The background idle
    reaper is a safety net for investigations that never call release() cleanly (a worker crash,
    an unhandled exception escaping the caller's own cleanup) - it destroys any sandbox unused
    for longer than idle_timeout_seconds.
    """

    def __init__(
        self,
        socket_root: str,
        image: str = IMAGE_NAME,
        mem_limit: str = "512m",
        nano_cpus: int = 1_000_000_000,
        idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS,
        health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
        reap_interval_seconds: int = DEFAULT_REAP_INTERVAL_SECONDS,
        start_reaper: bool = True,
    ):
        self.socket_root = os.path.abspath(socket_root)
        os.makedirs(self.socket_root, exist_ok=True)
        self.image = image
        self.mem_limit = mem_limit
        self.nano_cpus = nano_cpus
        self.idle_timeout_seconds = idle_timeout_seconds
        self.health_timeout_seconds = health_timeout_seconds
        self.reap_interval_seconds = reap_interval_seconds

        self._sandboxes: dict[str, _SandboxHandle] = {}
        self._global_lock = threading.Lock()
        self._client = None

        self._reaper_stop = threading.Event()
        self._reaper_thread = None
        if start_reaper:
            self._reaper_thread = threading.Thread(
                target=self._reap_loop, name="sandbox-idle-reaper", daemon=True,
            )
            self._reaper_thread.start()

        logger.info(
            "SandboxManager started (image=%s, idle_timeout=%ds, reap_interval=%ds, socket_root=%s)",
            image, idle_timeout_seconds, reap_interval_seconds, self.socket_root,
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
            logger.debug("sandbox image cached, check took %.3fs", time.perf_counter() - start)
        except ImageNotFound:
            logger.info("sandbox image not found - building from %s (this only happens once)", _SANDBOX_DIR)
            self.client.images.build(path=_SANDBOX_DIR, tag=self.image, rm=True)
            logger.info("sandbox image built in %.3fs", time.perf_counter() - start)

    # ------------------------------------------------------------------ creation / reuse

    def get_or_create(self, investigation_id: str) -> SandboxClient:
        """Every path through here logs WHY it did what it did (reused / recreated because
        unhealthy / created because none existed) - see _liveness's reason string and the three
        logger calls below - so "why did this take 2-3s" is always answerable from the logs
        alone, without having to guess between "first ever call for this investigation" and
        "the container died and had to be rebuilt"."""
        try:
            validate_segment(investigation_id, "investigation_id")
        except InvalidArtifactIdError as exc:
            raise SandboxManagerError(str(exc)) from exc

        with self._global_lock:
            handle = self._sandboxes.get(investigation_id)

            if handle is None:
                logger.info(
                    "sandbox missing: no cached sandbox for investigation=%s yet - creating one now "
                    "(first run_python call, or an earlier pre-warm that hasn't landed yet - the "
                    "lock this method holds means only ONE of those actually calls docker.run)",
                    investigation_id,
                )
                handle = self._create_sandbox(investigation_id)
                self._sandboxes[investigation_id] = handle
                return handle.client

            alive, reason = self._liveness(handle)
            if alive:
                logger.info(
                    "sandbox reuse: investigation=%s container=%s (age=%.0fs, idle=%.0fs)",
                    investigation_id, handle.container.short_id,
                    time.time() - handle.created_at, time.time() - handle.last_used_at,
                )
                handle.last_used_at = time.time()
                return handle.client

            logger.warning(
                "sandbox unhealthy: investigation=%s container=%s reason=%r (age=%.0fs) - "
                "destroying and recreating",
                investigation_id, handle.container.short_id, reason, time.time() - handle.created_at,
            )
            self._destroy_handle(handle)
            self._sandboxes.pop(investigation_id, None)

            handle = self._create_sandbox(investigation_id)
            self._sandboxes[investigation_id] = handle
            return handle.client

    def _liveness(self, handle: _SandboxHandle) -> tuple[bool, str | None]:
        """(is_alive, reason_if_not) - the reason string is purely for the "sandbox unhealthy"
        log line above, so a dead container's cause (Docker daemon lost track of it vs. the
        container process itself exited/crashed vs. some other status) is visible without
        needing to go dig through `docker ps -a`/`docker logs` by hand."""
        try:
            handle.container.reload()
        except Exception as exc:
            return False, f"container.reload() failed (daemon lost track of it?): {exc}"
        if handle.container.status != "running":
            return False, f"container status is {handle.container.status!r}, not 'running'"
        return True, None

    def _create_sandbox(self, investigation_id: str) -> _SandboxHandle:
        logger.info("sandbox creation: starting new container for investigation=%s", investigation_id)
        self.ensure_image()

        socket_filename = f"{investigation_id}.sock"
        host_socket_path = os.path.join(self.socket_root, socket_filename)
        if os.path.exists(host_socket_path):
            logger.warning(
                "sandbox creation: removing stale socket file %s before starting new container",
                host_socket_path,
            )
            try:
                os.remove(host_socket_path)
            except OSError:
                logger.exception("failed to remove stale socket file %s", host_socket_path)

        create_start = time.perf_counter()
        container = self.client.containers.run(
            self.image,
            detach=True,
            # No network access, same as the old one-shot container - a Unix domain socket over
            # a shared filesystem volume needs no network namespace at all, so this security
            # property carries over unchanged.
            network_disabled=True,
            mem_limit=self.mem_limit,
            nano_cpus=self.nano_cpus,
            volumes={
                PARQUET_VOLUME_NAME: {"bind": "/data/parquet", "mode": "rw"},
                SANDBOX_SOCKET_VOLUME_NAME: {"bind": SANDBOX_SOCKET_CONTAINER_MOUNT, "mode": "rw"},
            },
            environment={"SANDBOX_ID": investigation_id},
            labels={"dataanalyzer.investigation_id": investigation_id},
        )
        create_s = time.perf_counter() - create_start
        logger.info(
            "sandbox creation: container %s created+started for investigation=%s in %.3fs",
            container.short_id, investigation_id, create_s,
        )

        client = SandboxClient(host_socket_path)
        try:
            self._wait_for_health(client, container, investigation_id)
        except Exception:
            try:
                container.remove(force=True)
            except Exception:
                pass
            raise

        return _SandboxHandle(investigation_id, container, client, host_socket_path)

    def _wait_for_health(self, client: SandboxClient, container, investigation_id: str) -> None:
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
                        f"sandbox container for investigation {investigation_id} exited before "
                        f"becoming healthy (status={container.status}): {logs}"
                    )
                client.health(timeout=1.5)
                logger.info(
                    "UDS connection established: investigation=%s container=%s socket=%s "
                    "(after %d attempt(s), %.2fs)",
                    investigation_id, container.short_id, client.socket_path,
                    attempt, self.health_timeout_seconds - (deadline - time.perf_counter()),
                )
                return
            except SandboxManagerError:
                raise
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)

        raise SandboxManagerError(
            f"sandbox for investigation {investigation_id} did not become healthy within "
            f"{self.health_timeout_seconds}s ({attempt} attempt(s)): {last_error}"
        )

    # ------------------------------------------------------------------ execution

    def execute(
        self, investigation_id: str, code: str, tables: dict, workspace_id: str,
        timeout_seconds: float = None,
    ) -> dict:
        client = self.get_or_create(investigation_id)
        handle = self._sandboxes.get(investigation_id)
        lock = handle.lock if handle is not None else threading.Lock()

        with lock:
            t0 = time.perf_counter()
            try:
                result = client.execute(code, tables, workspace_id, timeout_seconds=timeout_seconds)
            except SandboxClientError as exc:
                logger.warning(
                    "execution failed for investigation=%s: %s - destroying sandbox so the next "
                    "call gets a fresh container instead of repeating the same failure",
                    investigation_id, exc,
                )
                self.release(investigation_id)
                return {"stdout": "", "saved": [], "error": str(exc)}
            finally:
                if handle is not None:
                    handle.last_used_at = time.time()
            logger.info(
                "execution time: investigation=%s completed in %.1fms",
                investigation_id, (time.perf_counter() - t0) * 1000,
            )
            return result

    # ------------------------------------------------------------------ cleanup

    def release(self, investigation_id: str) -> None:
        """Called when an investigation ends (worker_service/tasks/investigation.py's finally
        block, and dashboard_refresh.py after its one-off sandbox call) - destroys that
        investigation's container/socket immediately rather than waiting for the idle reaper, so
        a busy worker doesn't accumulate one warm container per completed investigation
        indefinitely."""
        with self._global_lock:
            handle = self._sandboxes.pop(investigation_id, None)
        if handle is None:
            logger.debug("release: no sandbox cached for investigation=%s (already gone or never created)", investigation_id)
            return
        logger.info("investigation %s ended - releasing sandbox %s", investigation_id, handle.container.short_id)
        self._destroy_handle(handle)

    def _destroy_handle(self, handle: _SandboxHandle) -> None:
        try:
            handle.client.shutdown()
        except Exception:
            pass
        try:
            handle.container.remove(force=True)
            logger.info("sandbox cleanup: container %s removed (investigation=%s)",
                        handle.container.short_id, handle.investigation_id)
        except NotFound:
            pass
        except Exception:
            logger.exception("failed to remove sandbox container %s", handle.container.short_id)
        finally:
            handle.client.close()
        try:
            if os.path.exists(handle.socket_path):
                os.remove(handle.socket_path)
                logger.info("sandbox cleanup: removed socket file %s", handle.socket_path)
        except OSError:
            logger.exception("failed to remove socket file %s", handle.socket_path)

    def _reap_loop(self) -> None:
        logger.info("idle reaper started (interval=%ds, idle_timeout=%ds)",
                    self.reap_interval_seconds, self.idle_timeout_seconds)
        while not self._reaper_stop.wait(self.reap_interval_seconds):
            now = time.time()
            with self._global_lock:
                idle_ids = [
                    inv_id for inv_id, h in self._sandboxes.items()
                    if now - h.last_used_at > self.idle_timeout_seconds
                ]
            for inv_id in idle_ids:
                logger.info(
                    "idle timeout: investigation=%s idle beyond %ds - reaping",
                    inv_id, self.idle_timeout_seconds,
                )
                self.release(inv_id)

    def shutdown_all(self) -> None:
        """Called from worker_service's on_shutdown - releases every sandbox still cached
        (e.g. jobs that crashed before reaching their own cleanup) and stops the reaper thread so
        the process can exit cleanly."""
        logger.info("SandboxManager shutdown: releasing %d sandbox(es)", len(self._sandboxes))
        self._reaper_stop.set()
        with self._global_lock:
            ids = list(self._sandboxes.keys())
        for inv_id in ids:
            self.release(inv_id)


_manager_lock = threading.Lock()
_default_manager: "SandboxManager | None" = None


def get_manager(socket_root: str = None, **kwargs) -> SandboxManager:
    """Process-wide singleton - an arq worker is one process serving many jobs, and every job
    (and every PythonSandbox instance any of those jobs construct) needs to land on the SAME
    manager to actually get cache reuse/idle cleanup across calls. worker_service.worker.py's
    on_startup calls this once explicitly (with socket_root=engine_bootstrap.SANDBOX_SOCKET_ROOT)
    so the instance - and any non-default kwargs - are established before any job runs; later
    calls from PythonSandbox/investigation.py omit socket_root and just get that same instance
    back. Only used when running this codebase as a bare script/test WITHOUT going through
    worker.py's on_startup does socket_root's own fallback default below actually get used."""
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                root = socket_root or os.environ.get("SANDBOX_SOCKET_ROOT") or os.path.join(
                    _SANDBOX_DIR, "..", "data", "sandbox_sockets",
                )
                _default_manager = SandboxManager(socket_root=root, **kwargs)
    return _default_manager
