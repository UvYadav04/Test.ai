import logging

from sandbox.path_resolver import InvalidArtifactIdError, validate_segment
from sandbox.sandbox_manager import SandboxManagerError, get_manager

logger = logging.getLogger("tools.tabular.sandbox")


class SandboxExecutionError(RuntimeError):
    pass


class PythonSandbox:
    """Thin per-agent-instance wrapper around the shared sandbox pool.

    This does NOT own a dedicated sandbox - `session_id` is kept only as a tag for
    logging/attribution. Every run() call acquires whatever sandbox is idle in the shared
    pool, executes on it, and returns it; the pool (not this object) owns container
    lifecycle.
    """

    def __init__(
        self,
        root_dir: str,
        session_id: str = "default",
        timeout_seconds: int = 30,
        mem_limit: str = "512m",
        nano_cpus: int = 1_000_000_000,
        manager=None,
    ):
        self.root_dir = root_dir
        self.session_id = session_id
        self.timeout_seconds = timeout_seconds
        self._mem_limit = mem_limit
        self._nano_cpus = nano_cpus
        self._manager = manager

    @property
    def manager(self):
        if self._manager is None:
            self._manager = get_manager(mem_limit=self._mem_limit, nano_cpus=self._nano_cpus)
        return self._manager

    def run(self, code: str, tables: dict, workspace_id: str) -> dict:
        try:
            workspace_id = validate_segment(workspace_id, "workspace_id")
            container_tables = {
                table_name: validate_segment(file_id, f"file_id for table '{table_name}'")
                for table_name, file_id in tables.items()
            }
        except InvalidArtifactIdError as exc:
            raise SandboxExecutionError(str(exc)) from exc

        logger.info(
            "sandbox run: tag=%s workspace=%s tables=%d timeout=%ss",
            self.session_id, workspace_id, len(container_tables), self.timeout_seconds,
        )
        try:
            # manager.execute() retries internally against the pool (see
            # SandboxManager.MAX_EXECUTE_ATTEMPTS) and always returns an {"error": ...} dict
            # rather than raising once attempts are exhausted - this except clause is just a
            # defensive fallback for anything that still manages to raise before that loop
            # starts (e.g. an invalid workspace_id failing validation, or the pool being
            # exhausted and every acquire() attempt timing out).
            return self.manager.execute(
                code, container_tables, workspace_id,
                timeout_seconds=self.timeout_seconds, tag=self.session_id,
            )
        except SandboxManagerError as exc:
            logger.warning("sandbox run: tag=%s manager.execute raised: %s", self.session_id, exc)
            raise SandboxExecutionError(str(exc)) from exc
