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

PARQUET_VOLUME_NAME = os.environ.get("PARQUET_VOLUME_NAME", "dataanalyzer_parquet_data")

SANDBOX_SOCKET_VOLUME_NAME = os.environ.get("SANDBOX_SOCKET_VOLUME_NAME", "dataanalyzer_sandbox_sockets")
SANDBOX_SOCKET_CONTAINER_MOUNT = "/shared"

# 5 minutes - a sandbox is now scoped and kept warm per CHAT (see get_or_create's session_id
# param, and worker_service/tasks/investigation.py which passes chat_id as that id), not per
# investigation/turn, so this is genuinely "how long can a chat sit idle before we give the
# container back" rather than a single request's timeout budget.
DEFAULT_IDLE_TIMEOUT_SECONDS = int(os.environ.get("SANDBOX_IDLE_TIMEOUT_SECONDS", "300"))
DEFAULT_HEALTH_TIMEOUT_SECONDS = float(os.environ.get("SANDBOX_HEALTH_TIMEOUT_SECONDS", "15"))
DEFAULT_REAP_INTERVAL_SECONDS = int(os.environ.get("SANDBOX_REAP_INTERVAL_SECONDS", "60"))


class SandboxManagerError(RuntimeError):
    pass


class _SandboxHandle:
    __slots__ = ("session_id", "container", "client", "socket_path", "created_at", "last_used_at", "lock")

    def __init__(self, session_id: str, container, client: SandboxClient, socket_path: str):
        self.session_id = session_id
        self.container = container
        self.client = client
        self.socket_path = socket_path
        self.created_at = time.time()
        self.last_used_at = self.created_at
        self.lock = threading.Lock()


class SandboxManager:
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
        # user_id -> the session_id (chat_id) that user's one persistent sandbox currently
        # belongs to. Only used to enforce "one warm sandbox per user at a time" - see
        # get_or_create's user_id param / _evict_other_sessions_for_user. Deliberately NOT
        # reverse-cleaned when a session is reaped/released some other way (idle timeout, an
        # unhealthy-container recreate): it just self-corrects the next time that user sends a
        # message, and release() on an already-gone session_id is already a safe no-op.
        self._active_session_by_user: dict[str, str] = {}
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

    def get_or_create(self, session_id: str, user_id: str | None = None) -> SandboxClient:
        """`session_id` is chat_id in practice (see module docstring at the top of this file) -
        the sandbox persists warm across every message in that chat, not just within one
        investigation/turn.

        `user_id`, when given, enforces "one warm sandbox per user at a time": if this user's
        previous message was in a DIFFERENT session_id (chat), that chat's sandbox is released
        first. Pass user_id only from the ONE call site that represents "a user just started
        interacting with this chat" (the pre-warm at the top of run_investigation) - internal
        reuse calls (e.g. from execute()) should leave it None so they don't re-trigger eviction
        mid-investigation."""
        try:
            validate_segment(session_id, "session_id")
        except InvalidArtifactIdError as exc:
            raise SandboxManagerError(str(exc)) from exc

        if user_id is not None:
            self._evict_other_sessions_for_user(user_id, session_id)

        with self._global_lock:
            handle = self._sandboxes.get(session_id)

            if handle is None:
                logger.info(
                    "sandbox missing: no cached sandbox for session=%s yet - creating one now "
                    "(first run_python call in this chat, or an earlier pre-warm that hasn't "
                    "landed yet - the lock this method holds means only ONE of those actually "
                    "calls docker.run)",
                    session_id,
                )
                handle = self._create_sandbox(session_id)
                self._sandboxes[session_id] = handle
                return handle.client

            alive, reason = self._liveness(handle)
            if alive:
                logger.info(
                    "sandbox reuse: session=%s container=%s (age=%.0fs, idle=%.0fs)",
                    session_id, handle.container.short_id,
                    time.time() - handle.created_at, time.time() - handle.last_used_at,
                )
                handle.last_used_at = time.time()
                return handle.client

            logger.warning(
                "sandbox unhealthy: session=%s container=%s reason=%r (age=%.0fs) - "
                "destroying and recreating",
                session_id, handle.container.short_id, reason, time.time() - handle.created_at,
            )
            self._destroy_handle(handle)
            self._sandboxes.pop(session_id, None)

            handle = self._create_sandbox(session_id)
            self._sandboxes[session_id] = handle
            return handle.client

    def _evict_other_sessions_for_user(self, user_id: str, session_id: str) -> None:
        with self._global_lock:
            previous = self._active_session_by_user.get(user_id)
            self._active_session_by_user[user_id] = session_id
        if previous is not None and previous != session_id:
            logger.info(
                "sandbox eviction: user=%s switched from session=%s to session=%s - releasing "
                "the previous session's sandbox",
                user_id, previous, session_id,
            )
            # Outside the lock above - release() takes _global_lock itself, and it isn't
            # reentrant.
            self.release(previous)

    def _liveness(self, handle: _SandboxHandle) -> tuple[bool, str | None]:
        try:
            handle.container.reload()
        except Exception as exc:
            return False, f"container.reload() failed (daemon lost track of it?): {exc}"
        if handle.container.status != "running":
            return False, f"container status is {handle.container.status!r}, not 'running'"
        return True, None

    def _create_sandbox(self, session_id: str) -> _SandboxHandle:
        logger.info("sandbox creation: starting new container for session=%s", session_id)
        self.ensure_image()

        socket_filename = f"{session_id}.sock"
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
            network_disabled=True,
            mem_limit=self.mem_limit,
            nano_cpus=self.nano_cpus,
            volumes={
                PARQUET_VOLUME_NAME: {"bind": "/data/parquet", "mode": "rw"},
                SANDBOX_SOCKET_VOLUME_NAME: {"bind": SANDBOX_SOCKET_CONTAINER_MOUNT, "mode": "rw"},
            },
            environment={"SANDBOX_ID": session_id},
            labels={"dataanalyzer.session_id": session_id},
        )
        create_s = time.perf_counter() - create_start
        logger.info(
            "sandbox creation: container %s created+started for session=%s in %.3fs",
            container.short_id, session_id, create_s,
        )

        client = SandboxClient(host_socket_path)
        try:
            self._wait_for_health(client, container, session_id)
        except Exception:
            try:
                container.remove(force=True)
            except Exception:
                pass
            raise

        return _SandboxHandle(session_id, container, client, host_socket_path)

    def _wait_for_health(self, client: SandboxClient, container, session_id: str) -> None:
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
                        f"sandbox container for session {session_id} exited before "
                        f"becoming healthy (status={container.status}): {logs}"
                    )
                client.health(timeout=1.5)
                logger.info(
                    "UDS connection established: session=%s container=%s socket=%s "
                    "(after %d attempt(s), %.2fs)",
                    session_id, container.short_id, client.socket_path,
                    attempt, self.health_timeout_seconds - (deadline - time.perf_counter()),
                )
                return
            except SandboxManagerError:
                raise
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)

        raise SandboxManagerError(
            f"sandbox for session {session_id} did not become healthy within "
            f"{self.health_timeout_seconds}s ({attempt} attempt(s)): {last_error}"
        )

    def execute(
        self, session_id: str, code: str, tables: dict, workspace_id: str,
        timeout_seconds: float = None,
    ) -> dict:
        # No user_id here on purpose - this is a reuse/continuation of an already-pre-warmed
        # session (or a same-turn first call), not a new "user started interacting" event, so it
        # must not re-trigger the one-sandbox-per-user eviction (see get_or_create's docstring).
        client = self.get_or_create(session_id)
        handle = self._sandboxes.get(session_id)
        lock = handle.lock if handle is not None else threading.Lock()

        with lock:
            t0 = time.perf_counter()
            try:
                result = client.execute(code, tables, workspace_id, timeout_seconds=timeout_seconds)
            except SandboxClientError as exc:
                logger.warning(
                    "execution failed for session=%s: %s - destroying sandbox so the next "
                    "call gets a fresh container instead of repeating the same failure",
                    session_id, exc,
                )
                self.release(session_id)
                return {"stdout": "", "saved": [], "error": str(exc)}
            finally:
                if handle is not None:
                    handle.last_used_at = time.time()
            logger.info(
                "execution time: session=%s completed in %.1fms",
                session_id, (time.perf_counter() - t0) * 1000,
            )
            return result

    def release(self, session_id: str) -> None:
        with self._global_lock:
            handle = self._sandboxes.pop(session_id, None)
        if handle is None:
            logger.debug("release: no sandbox cached for session=%s (already gone or never created)", session_id)
            return
        logger.info("session %s ended/evicted/idled out - releasing sandbox %s", session_id, handle.container.short_id)
        self._destroy_handle(handle)

    def _destroy_handle(self, handle: _SandboxHandle) -> None:
        try:
            handle.client.shutdown()
        except Exception:
            pass
        try:
            handle.container.remove(force=True)
            logger.info("sandbox cleanup: container %s removed (session=%s)",
                        handle.container.short_id, handle.session_id)
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
                    session_id for session_id, h in self._sandboxes.items()
                    if now - h.last_used_at > self.idle_timeout_seconds
                ]
            for session_id in idle_ids:
                logger.info(
                    "idle timeout: session=%s idle beyond %ds (no message in this chat) - reaping",
                    session_id, self.idle_timeout_seconds,
                )
                self.release(session_id)

    def shutdown_all(self) -> None:
        logger.info("SandboxManager shutdown: releasing %d sandbox(es)", len(self._sandboxes))
        self._reaper_stop.set()
        with self._global_lock:
            ids = list(self._sandboxes.keys())
        for session_id in ids:
            self.release(session_id)


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
