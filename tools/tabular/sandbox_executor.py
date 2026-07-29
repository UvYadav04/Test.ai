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
        investigation_id: str = "default",
        timeout_seconds: int = 30,
        mem_limit: str = "512m",
        nano_cpus: int = 1_000_000_000,
        manager=None,
    ):
        self.root_dir = root_dir
        self.investigation_id = investigation_id
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

        logger.debug(
            "sandbox run: investigation=%s workspace=%s tables=%d",
            self.investigation_id, workspace_id, len(container_tables),
        )
        try:
            return self.manager.execute(
                self.investigation_id, code, container_tables, workspace_id,
                timeout_seconds=self.timeout_seconds,
            )
        except SandboxManagerError as exc:
            raise SandboxExecutionError(str(exc)) from exc
