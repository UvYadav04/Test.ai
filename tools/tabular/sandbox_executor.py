"""Host-side driver for running model-generated pandas/DuckDB code inside an isolated Docker
sandbox. PythonSandbox's public API (construct with a root_dir, call .run(code, tables,
workspace_id)) is UNCHANGED from when every call spun up its own one-shot container - see git
history for that version. What changed is entirely internal: instead of creating+destroying a
fresh container per call, this now asks the process-wide SandboxManager (sandbox_manager.py) for
the ONE persistent, long-lived sandbox container that belongs to this investigation
(investigation_id, passed in at construction - see TabularTools.__init__) and sends this call's
code to it over HTTP-over-Unix-Domain-Socket (sandbox_client.py) instead of over a container's
stdin/manifest-file/exit-code lifecycle. The sandbox itself has no network access, a capped
memory/CPU allowance, and a hard wall-clock timeout, exactly as before - only whatever the
sandbox explicitly returns (stdout, capped; describe()/preview()/save() outputs) ever leaves it,
never the full DataFrames.

Tables cross this boundary as file_ids, never as paths: `run(tables={table_name: file_id})`
validates each id and workspace_id (the actual path is derived independently inside the
container by ExecutionEngine via get_sandbox_path(workspace_id, file_id) - see
sandbox/path_resolver.py and sandbox/execution_engine.py, which do this the same way this file's
predecessor's runner.py counterpart always did).

See sandbox/sandbox_manager.py's own docstring for the docker-outside-of-docker /
named-Docker-volume notes that used to live in this file - they now apply to both the parquet
volume AND the new sandbox-socket volume, and are documented once, there, instead of twice.
"""
import logging

from sandbox.path_resolver import InvalidArtifactIdError, validate_segment
from sandbox.sandbox_manager import SandboxManagerError, get_manager

logger = logging.getLogger("tools.tabular.sandbox")


class SandboxExecutionError(RuntimeError):
    pass


class PythonSandbox:
    """One instance is constructed per TabularTools instance (i.e. per invoke_tabular_agent call
    - see agents/tabular/agent.py), which is cheap: unlike before, constructing this no longer
    implies constructing a container. The actual container is looked up/created lazily, keyed by
    investigation_id, the first time .run() is actually called - see SandboxManager.get_or_create.
    Every OTHER PythonSandbox instance created for the SAME investigation_id (e.g. a second
    invoke_tabular_agent call later in the same investigation) resolves to that identical
    container via the shared SandboxManager singleton."""

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
        """In normal operation `self._manager` is never None here - TabularTools passes the real
        SandboxManager instance in explicitly (threaded all the way down from
        ctx["sandbox_manager"], built once in worker.py's on_startup - see
        OrchestratorAgent.__init__'s note on why). The get_manager() fallback below only fires
        for a PythonSandbox built standalone (a script, a test) without that chain - do NOT rely
        on it in worker_service's real request path: get_manager() is a module-level singleton
        keyed by which import path reached it, and `analyzerEngine.sandbox.sandbox_manager`
        (worker_service's own imports) vs bare `sandbox.sandbox_manager` (this module's import
        two lines up) are two DIFFERENT entries in sys.modules, each with its own independent
        singleton instance - see sandbox_manager.get_manager's docstring. Relying on this
        fallback from both sides is exactly what silently created two disconnected sandboxes
        (one pre-warmed, one used for real) before explicit injection was wired through."""
        if self._manager is None:
            self._manager = get_manager(mem_limit=self._mem_limit, nano_cpus=self._nano_cpus)
        return self._manager

    def run(self, code: str, tables: dict, workspace_id: str) -> dict:
        """tables: {table_name: file_id} - NOT a path. Each file_id is expected to already be a
        real parquet artifact under this workspace (the caller is responsible for that - e.g.
        TabularTools only ever passes file_ids it already validated as assigned/known).
        validate_segment below is the same containment guard this method always applied, before
        handing workspace_id/tables off to whichever sandbox (one-shot before, persistent now)
        actually resolves them to a path."""
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
