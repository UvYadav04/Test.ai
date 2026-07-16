import re
import uuid
from datetime import datetime, timezone

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.logger import get_agent_logger, log_event
from agents.orchestrator.config import SYSTEM_MESSAGE, get_model_config
from llm_provider import LLMProvider
from tools.orchestrator.models import InvestigationState, OrchestratorResult
from tools.orchestrator.orchestrator_tools import OrchestratorTools

_DELIVERABLE_TOOLS = {"generate_csv", "generate_markdown_report", "generate_dashboard"}


class OrchestratorAgent:
    def __init__(self, catalog, vector_store=None, reranker=None, memory=None, storage=None):
        self.logger = get_agent_logger("orchestrator_agent")
        model_config = get_model_config()
        client = LLMProvider(model_config["provider"], fallback_provider="groq").get_client(model_config["model"])

        self.tools = OrchestratorTools(
            catalog, state=None, vector_store=vector_store, reranker=reranker, memory=memory, storage=storage
        )

        self.agent = AssistantAgent(
            name="orchestrator_agent",
            model_client=client,
            tools=[
                self.tools.get_current_date,
                self.tools.recall_user_info,
                self.tools.store_user_info,
                self.tools.list_files,
                self.tools.search_files,
                self.tools.get_file_details,
                self.tools.list_tables,
                self.tools.list_file_formats,
                self.tools.generate_hypotheses,
                self.tools.invoke_tabular_agent,
                self.tools.invoke_document_agent,
                self.tools.generate_csv,
                self.tools.generate_markdown_report,
                self.tools.generate_dashboard,
            ],
            system_message=SYSTEM_MESSAGE,
            reflect_on_tool_use=False,
            max_tool_iterations=25,
        )

    async def run(self, objective: str, workspace_id: str = "default", constraints: dict = None) -> OrchestratorResult:
        await self.agent.on_reset(CancellationToken())

        constraints = constraints or {}
        self.tools.workspace_id = workspace_id
        self.tools.state = InvestigationState(
            session_id=uuid.uuid4().hex[:12],
            objective=objective,
            constraints=constraints,
        )

        task = (
            f"Objective: {objective}\n"
            f"Workspace: {workspace_id}\n"
            f"Constraints: {constraints}\n\n"
            f"{self._context_brief()}"
        )
        self.logger.info("objective sent to agent: %s", task)

        transcript = []
        final_text = ""
        async for event in self.agent.run_stream(task=task):
            if not hasattr(event, "messages"):
                log_event(self.logger, event)
                line = self._transcript_line(event)
                if line:
                    transcript.append(line)
                if type(event).__name__ == "TextMessage" and event.source == self.agent.name:
                    final_text = event.content

        self.logger.info("final reply: %s", final_text)
        return OrchestratorResult(
            final_answer=final_text,
            artifact_refs=self._collect_artifact_refs(transcript),
            open_questions=self.tools.state.open_questions,
        )

    def _collect_artifact_refs(self, transcript: list) -> list:
        """Real artifact paths only: whatever the delegated Tabular/Document agents already
        reported (from their own tool results, not an LLM's transcription of them), plus any
        deliverable file path a generate_csv/generate_markdown_report/generate_dashboard call
        actually returned."""
        refs = []
        for event in self.tools.state.completed_tasks:
            for ref in getattr(event.result, "artifact_refs", None) or []:
                if ref not in refs:
                    refs.append(ref)

        pattern = re.compile(r"^RESULT (\w+) -> (.+)$")
        for line in transcript:
            for sub_line in line.split("\n"):
                match = pattern.match(sub_line)
                if match and match.group(1) in _DELIVERABLE_TOOLS:
                    path = match.group(2).strip().strip("'\"")
                    if path and path not in refs:
                        refs.append(path)
        return refs

    def _context_brief(self, max_files: int = 40, max_columns: int = 25) -> str:
        """Precompute what get_current_date/recall_user_info/list_files would return and hand
        it to the agent directly in the task message, instead of making it spend its first 2-4
        tool calls (each a full model round trip) re-fetching things we already know here for
        free. The agent still has all these tools available for anything beyond this - a fuzzy
        name match, a workspace with more files than shown, or re-checking something mid-run."""
        now = datetime.now(timezone.utc)
        lines = [f"Today's date: {now.date().isoformat()} ({now.strftime('%A')}, UTC)."]

        user_info = self.tools.memory.recall_all()
        if user_info:
            lines.append("Known standing user preferences/facts (from recall_user_info):")
            lines.extend(f"- {fact}" for fact in user_info)
        else:
            lines.append("No standing user preferences/facts saved yet.")

        entries = self.tools.catalog.all()
        if not entries:
            lines.append("Workspace files: none uploaded yet.")
        else:
            shown = entries[:max_files]
            lines.append(f"Workspace files ({len(entries)} total, from list_files):")
            for e in shown:
                detail = f"- {e.filename} [file_id={e.file_id}, type={e.file_type}"
                if e.row_count is not None:
                    detail += f", {e.row_count} rows"
                if e.page_count is not None:
                    detail += f", {e.page_count} pages"
                detail += "]"
                if e.columns:
                    cols = e.columns[:max_columns]
                    col_str = ", ".join(cols)
                    if len(e.columns) > max_columns:
                        col_str += f", ... (+{len(e.columns) - max_columns} more)"
                    detail += f" columns: {col_str}"
                lines.append(detail)
            if len(entries) > max_files:
                lines.append(
                    f"... and {len(entries) - max_files} more files not shown here - call "
                    "list_files/search_files if you need to see them."
                )

        return "\n".join(lines)

    @staticmethod
    def _transcript_line(event) -> str:
        event_type = type(event).__name__
        if event_type == "ToolCallRequestEvent":
            return "\n".join(f"CALL {call.name}({call.arguments})" for call in event.content)
        if event_type == "ToolCallExecutionEvent":
            return "\n".join(f"RESULT {res.name} -> {res.content}" for res in event.content)
        return ""
