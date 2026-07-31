import logging

from sandbox.path_resolver import InvalidArtifactIdError, validate_segment
from sandbox.sandbox_manager import SandboxManagerError, get_manager

logger = logging.getLogger("tools.tabular.sandbox")


class SandboxExecutionError(RuntimeError):
    pass


class PythonSandbox:
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
            "sandbox run: session=%s workspace=%s tables=%d timeout=%ss",
            self.session_id, workspace_id, len(container_tables), self.timeout_seconds,
        )
        try:
            # manager.execute() itself retries internally (see SandboxManager.MAX_EXECUTE_ATTEMPTS)
            # and always returns an {"error": ...} dict rather than raising once it's exhausted
            # those attempts - this except clause is just a defensive fallback for anything that
            # still manages to raise SandboxManagerError before that loop starts (e.g. an invalid
            # session_id failing validation).
            return self.manager.execute(
                self.session_id, code, container_tables, workspace_id,
                timeout_seconds=self.timeout_seconds,
            )
        except SandboxManagerError as exc:
            logger.warning("sandbox run: session=%s manager.execute raised: %s", self.session_id, exc)
            raise SandboxExecutionError(str(exc)) from exc
