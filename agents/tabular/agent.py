import json
import os
import re
import time

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.events import make_tool_event_translator
from agents.final_answer import split_follow_up_questions
from agents.logger import get_agent_logger, log_event
from agents.tabular.config import get_model_config, get_system_message
from agents.thread_context import thread_context_brief
from agents.timing import ToolCallTimer
from llm_provider import LLMProvider
from sandbox.path_resolver import InvalidArtifactIdError, get_parquet_path
from tools.orchestrator.models import TabularFindings
from tools.tabular.tabular_tools import TabularTools


class TabularAgent:
    def __init__(
        self, assigned_files: list, storage=None, workspace_id: str = "default",
        chat_id: str = "default", sandbox_manager=None, direct_route: bool = False,
        reports_dir: str = "data/reports", chart_capacity_checker=None,
    ):
        self.logger = get_agent_logger("tabular_agent")
        # chat_id (not investigation_id) keys the underlying sandbox - it now persists warm for
        # the whole chat (across turns), released only on chat switch or 5-minute idle, not torn
        # down at the end of every single investigation. See sandbox/sandbox_manager.py.
        self.tools = TabularTools(
            assigned_files, storage=storage, workspace_id=workspace_id,
            chat_id=chat_id, sandbox_manager=sandbox_manager,
            reports_dir=reports_dir, chart_capacity_checker=chart_capacity_checker,
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
            system_message=get_system_message(direct_route),
            reflect_on_tool_use=False,
            max_tool_iterations=10,
        )
        self.last_transform_script: str | None = None
        self.last_transform_file_ids: list = []

    async def run(
        self, objective: str, constraints: dict = None, on_event=None, thread_context: dict = None,
    ) -> TabularFindings:
        await self.agent.on_reset(CancellationToken())
        self.last_transform_script = None
        self.last_transform_file_ids = []
        self.tools.saved_artifacts = {}
        self.tools.charts_created = []

        constraints = constraints or {}
        allowed_files = self.tools.list_allowed_files()
        # thread_context matters most when this agent was DIRECT-routed (see run_investigation's
        # direct_route branch) - the Orchestrator never even gets involved, so this is the ONLY
        # place "show me a chart of THIS data" (referring to a file/result from an earlier turn)
        # can get resolved. When the Orchestrator invoked this agent instead, it's already folded
        # earlier turns into its own objective text, so this is usually redundant there - never
        # harmful either way.
        task = (
            f"Objective: {objective}\n"
            f"Assigned files - use these exact file_id/table_name values, do not guess or "
            f"invent others, and you do not need to call list_allowed_files again unless you "
            f"want to re-check them: {allowed_files}\n"
            f"Constraints: {constraints}\n\n"
            f"{thread_context_brief(thread_context)}"
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
        chart_locations = [entry["location"] for entry in self.tools.charts_created]
        # Only DIRECT_SYSTEM_MESSAGE actually asks for a FOLLOW_UP_QUESTIONS section (see
        # agents/tabular/config.py) - when this agent was delegated BY the Orchestrator instead,
        # final_text has no marker, so this is a harmless no-op there (returns final_text
        # unchanged, empty question list).
        summary, follow_up_questions = split_follow_up_questions(final_text)
        return TabularFindings(
            summary=summary,
            artifact_refs=real_refs + chart_locations,
            artifact_metadata=self._artifact_metadata(real_refs),
            charts=list(self.tools.charts_created),
            follow_up_questions=follow_up_questions,
        )

    def _artifact_metadata(self, real_refs: list) -> dict:
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

    _FRIENDLY_TOOL_NAMES = {
        "list_allowed_files": "Listing files",
        "run_python": "Executing a Python script",
        "create_visualizations": "Generating visualizations",
    }

    _translate_event = staticmethod(make_tool_event_translator(_FRIENDLY_TOOL_NAMES))

    @staticmethod
    def _extract_refs(transcript: list, key: str) -> list:
        text = "\n".join(transcript)
        pattern = rf"['\"]{re.escape(key)}['\"]\s*:\s*['\"]([^'\"]+)['\"]"
        refs = []
        for match in re.findall(pattern, text):
            if match and match not in refs:
                refs.append(match)
        return refs
