import json
import os
import re
import time

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.events import make_tool_call_translator
from agents.logger import get_agent_logger, log_event
from agents.tabular.config import get_model_config, get_system_message
from agents.timing import ToolCallTimer
from llm_provider import LLMProvider
from sandbox.path_resolver import InvalidArtifactIdError, get_parquet_path
from tools.orchestrator.models import TabularFindings
from tools.tabular.tabular_tools import TabularTools


class TabularAgent:
    def __init__(
        self, assigned_files: list, storage=None, workspace_id: str = "default",
        investigation_id: str = "default", sandbox_manager=None, direct_route: bool = False,
        reports_dir: str = "data/reports",
    ):
        self.logger = get_agent_logger("tabular_agent")
        self.tools = TabularTools(
            assigned_files, storage=storage, workspace_id=workspace_id,
            investigation_id=investigation_id, sandbox_manager=sandbox_manager,
            reports_dir=reports_dir,
        )
        model_config = get_model_config()
        client = LLMProvider(model_config["provider"]).get_client(model_config["model"])

        self.agent = AssistantAgent(
            name="tabular_agent",
            model_client=client,
            tools=[
                self.tools.list_allowed_files,
                self.tools.run_python,
                self.tools.create_visualizations,
            ],
            # direct_route=True when the controller routed straight here (bypassing the
            # Orchestrator) - see worker_service/tasks/investigation.py's _run_tabular_direct
            # and agents/tabular/config.py's DIRECT_ROUTE_ADDENDUM for what that changes.
            system_message=get_system_message(direct_route),
            reflect_on_tool_use=False,
            max_tool_iterations=10,
        )
        # Populated by run() below from the LAST run_python call this agent made (if any) -
        # NOT part of the TabularFindings returned to the orchestrator LLM, deliberately.
        # These exist so a caller that already holds a reference to this TabularAgent
        # instance (OrchestratorTools.invoke_tabular_agent, for a real-time dashboard) can
        # read the actual code/file_ids off the object after run() completes, the same
        # "pull it from the real tool call, never trust the LLM to retype it" spirit as
        # _extract_refs() below - just reaching one step further back, to the call
        # arguments instead of the call result. Keeping it off TabularFindings means every
        # OTHER invoke_tabular_agent call (the vast majority, which aren't building a
        # real-time dashboard) never pays for a potentially-large code blob riding along
        # in the orchestrator's context.
        self.last_transform_script: str | None = None
        self.last_transform_file_ids: list = []

    async def run(self, objective: str, constraints: dict = None, on_event=None) -> TabularFindings:
        """`on_event`, if given, is an `async def on_event(event: dict) -> None` -
        forwarded here from OrchestratorTools.invoke_tabular_agent so this
        agent's OWN tool calls (run_python, list_allowed_files) also surface
        on the live activity panel, not just "Assigning an agent" with
        nothing in between until it returns."""
        await self.agent.on_reset(CancellationToken())
        self.last_transform_script = None
        self.last_transform_file_ids = []
        # In practice this TabularTools instance is always fresh per run() (OrchestratorTools.
        # invoke_tabular_agent constructs a new TabularAgent every call) so these are already
        # empty - reset explicitly anyway for symmetry with the above, in case that ever changes.
        self.tools.saved_artifacts = {}
        self.tools.charts_created = []

        constraints = constraints or {}
        allowed_files = self.tools.list_allowed_files()
        task = (
            f"Objective: {objective}\n"
            f"Assigned files - use these exact file_id/table_name values, do not guess or "
            f"invent others, and you do not need to call list_allowed_files again unless you "
            f"want to re-check them: {allowed_files}\n"
            f"Constraints: {constraints}"
        )
        self.logger.info("objective sent to agent: %s", task)

        run_start = time.perf_counter()
        tool_timer = ToolCallTimer(self.logger)
        transcript = []
        final_text = ""
        async for event in self.agent.run_stream(task=task):
            if not hasattr(event, "messages"):
                log_event(self.logger, event)
                tool_timer.record(event)
                self._capture_run_python_call(event)
                line = self._transcript_line(event)
                if line:
                    transcript.append(line)
                if type(event).__name__ == "TextMessage" and event.source == self.agent.name:
                    final_text = event.content
                if on_event is not None:
                    translated = self._translate_event(event)
                    if translated:
                        await on_event(translated)

        self.logger.info("tabular agent run took %.3fs", time.perf_counter() - run_start)
        real_refs = self._real_refs(self._extract_refs(transcript, "file_id"))
        # self.tools.charts_created was populated by create_visualizations() calls that already
        # rendered and saved each chart for real (see TabularTools) - unlike real_refs above,
        # there's no LLM-transcription trust issue to filter against here: a chart_created entry
        # only exists because TabularTools itself confirmed the source artifact was really
        # save()'d and then actually wrote the chart file, not because the model claimed so.
        # Each entry's "location" rides along in artifact_refs so worker_service's existing
        # artifact-persistence pipeline (_persist_artifacts) uploads it and creates a Chart doc
        # exactly like it already does for generate_csv/generate_markdown_report output.
        chart_locations = [entry["location"] for entry in self.tools.charts_created]
        return TabularFindings(
            summary=final_text,
            artifact_refs=real_refs + chart_locations,
            artifact_metadata=self._artifact_metadata(real_refs),
            charts=list(self.tools.charts_created),
        )

    def _artifact_metadata(self, real_refs: list) -> dict:
        """{file_id: {row_count, columns, dtypes, column_kinds, preview}} for every file_id in
        real_refs that this agent's own run_python calls actually save()'d this run - see
        TabularTools.saved_artifacts. A real_ref that isn't in there (shouldn't normally happen,
        since _real_refs only keeps ids confirmed to exist on disk) is simply omitted rather than
        raising - missing metadata for one artifact shouldn't take down the whole findings
        object."""
        metadata = {}
        for file_id in real_refs:
            entry = self.tools.saved_artifacts.get(file_id)
            if entry is None:
                continue
            metadata[file_id] = {
                "row_count": entry.get("row_count"),
                "columns": entry.get("columns"),
                "dtypes": entry.get("dtypes"),
                "column_kinds": entry.get("column_kinds"),
                "preview": entry.get("preview"),
            }
        return metadata

    def _real_refs(self, candidates: list) -> list:
        """_extract_refs regex-scans the FULL tool-result text for anything shaped like
        `"file_id": "..."` - which also matches a spurious source: if the model's own
        run_python code echoes save()'s return value (e.g. `print(save(df, name=...))` or
        `print({"file_id": fid})`), that same string ends up in stdout verbatim alongside the
        structured "saved" list entries. Unlike the old output_ref-based version of this check,
        there's no host-vs-container path ambiguity to resolve anymore - a file_id is the exact
        same string on both sides of the sandbox boundary (see sandbox/path_resolver.py), so
        there's only ever one candidate shape per save() call, not two. The remaining risk is a
        fabricated id the model typed without ever calling save() - this process runs
        host-side, so confirming the id resolves to a real file on disk is still a cheap,
        reliable filter for that."""
        return [ref for ref in candidates if self._artifact_exists(ref)]

    def _artifact_exists(self, file_id: str) -> bool:
        root_dir = getattr(self.tools, "root_dir", None)
        if not root_dir:
            return False
        try:
            path = get_parquet_path(root_dir, self.tools.workspace_id, file_id)
        except InvalidArtifactIdError:
            return False
        return os.path.isfile(path)

    def _capture_run_python_call(self, event) -> None:
        """Keep the LAST run_python call's real arguments (not a re-transcription) on this
        instance - see the note in __init__. If the agent calls run_python more than once
        while exploring, the last call wins on the assumption that's the one whose save()
        outputs it actually used for its final answer."""
        if type(event).__name__ != "ToolCallRequestEvent":
            return
        for call in event.content:
            if call.name != "run_python":
                continue
            try:
                args = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError):
                continue
            code = args.get("code")
            if code:
                self.last_transform_script = code
                self.last_transform_file_ids = args.get("file_ids") or []

    @staticmethod
    def _transcript_line(event) -> str:
        event_type = type(event).__name__
        if event_type == "ToolCallRequestEvent":
            return "\n".join(f"CALL {call.name}({call.arguments})" for call in event.content)
        if event_type == "ToolCallExecutionEvent":
            return "\n".join(f"RESULT {res.name} -> {res.content}" for res in event.content)
        return ""

    # Same style/rules as OrchestratorAgent's own map (agents/events.py) -
    # genuine plain-language labels, no "done" counterpart.
    _FRIENDLY_TOOL_NAMES = {
        "list_allowed_files": "Listing files",
        "run_python": "Executing a Python script",
        "create_visualizations": "Generating visualizations",
    }

    _translate_event = staticmethod(make_tool_call_translator(_FRIENDLY_TOOL_NAMES))

    @staticmethod
    def _extract_refs(transcript: list, key: str) -> list:
        """Pull real file_ids straight out of tool results (e.g. run_python's save() entries)
        instead of trusting an LLM to transcribe them - the sandbox already returns the exact
        id, so re-deriving it from a second model call is both an extra round trip and a
        chance to hallucinate or drop it."""
        text = "\n".join(transcript)
        pattern = rf"['\"]{re.escape(key)}['\"]\s*:\s*['\"]([^'\"]+)['\"]"
        refs = []
        for match in re.findall(pattern, text):
            if match and match not in refs:
                refs.append(match)
        return refs
