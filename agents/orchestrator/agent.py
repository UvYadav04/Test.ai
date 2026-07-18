import json
import uuid
from datetime import datetime, timezone

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.logger import get_agent_logger, log_event
from agents.orchestrator.config import FORMAT_SYSTEM_MESSAGE, SYSTEM_MESSAGE, get_model_config
from llm_provider import LLMProvider
from tools.orchestrator.models import InvestigationState, OrchestratorResult
from tools.orchestrator.orchestrator_tools import OrchestratorTools


class InvestigationCancelled(Exception):
    """Raised by OrchestratorAgent.run() when cancel_check() returns True
    between steps. Callers (worker_service) catch this specifically to mark
    the Investigation as cancelled rather than completed/failed - the
    partial InvestigationState up to that point is still attached via
    `.state` for logging/debugging."""

    def __init__(self, state: InvestigationState):
        super().__init__("investigation cancelled")
        self.state = state


class OrchestratorAgent:
    def __init__(
        self, catalog, vector_store=None, reranker=None, memory=None, storage=None,
        reports_dir: str = "data/reports",
    ):
        self.logger = get_agent_logger("orchestrator_agent")
        model_config = get_model_config()
        client = LLMProvider(model_config["provider"], fallback_provider="groq").get_client(model_config["model"])

        self.tools = OrchestratorTools(
            catalog, state=None, vector_store=vector_store, reranker=reranker, memory=memory, storage=storage,
            reports_dir=reports_dir,
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

        self.formatter = AssistantAgent(
            name="orchestrator_formatter",
            model_client=client,
            system_message=FORMAT_SYSTEM_MESSAGE,
        )

    async def run(
        self,
        objective: str,
        workspace_id: str = "default",
        constraints: dict = None,
        on_event=None,
        cancel_check=None,
    ) -> OrchestratorResult:
        """`on_event`, if given, is an `async def on_event(event: dict) -> None`
        called once per meaningful step (tool call requested/executed) plus
        once more right before formatting the final answer - see
        `_translate_event` for the event shapes. `cancel_check`, if given, is
        an `async def cancel_check() -> bool` polled between steps (never
        mid-tool-call/mid-LLM-call); returning True stops the loop cleanly
        and raises InvestigationCancelled instead of returning a result."""
        await self.agent.on_reset(CancellationToken())
        await self.formatter.on_reset(CancellationToken())

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
        stream = self.agent.run_stream(task=task)
        try:
            async for event in stream:
                if not hasattr(event, "messages"):
                    log_event(self.logger, event)
                    line = self._transcript_line(event)
                    if line:
                        transcript.append(line)
                    if on_event is not None:
                        translated = self._translate_event(event)
                        if translated:
                            await on_event(translated)

                if cancel_check is not None and await cancel_check():
                    await stream.aclose()
                    if on_event is not None:
                        await on_event({"type": "cancelled", "message": "Investigation cancelled."})
                    raise InvestigationCancelled(self.tools.state)
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

        print("transcript : ",transcript)

        if on_event is not None:
            await on_event({"type": "status", "message": "Finalizing answer..."})

        format_task = (
            f"Objective: {objective}\n"
            f"Investigation State:\n{self.tools.state.summary()}\n"
            "Tool activity:\n" + "\n".join(transcript) +
            "\nRespond with ONLY the final JSON now."
        )
        format_result = await self.formatter.run(task=format_task)
        raw = format_result.messages[-1].content
        self.logger.info("final reply: %s", raw)
        result = self._parse(raw)

        if on_event is not None:
            await on_event({
                "type": "answer",
                "message": result.final_answer,
                "data": {"confidence": result.confidence, "artifact_refs": result.artifact_refs},
            })

        return result

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
        if event_type == "TextMessage" and getattr(event, "source", None) != "user":
            # The orchestrator's own final natural-language reply - this is the
            # only place it appears in the stream. Without it, the formatter
            # (which never sees the raw agent, only this transcript) has
            # nothing to ground on when no tools were called, and fabricates
            # an answer instead of reusing the one already given.
            return f"AGENT SAYS: {event.content}"
        return ""

    # Human-readable labels for the homescreen "live activity" panel (see
    # project_documentation_and_claude_code_guide.md Section 6) - plain-
    # language status, not raw tool logs.
    _FRIENDLY_TOOL_NAMES = {
        "list_files": "Listing workspace files",
        "search_files": "Searching workspace files",
        "get_file_details": "Inspecting a file",
        "list_tables": "Listing tables extracted from documents",
        "list_file_formats": "Checking available file types",
        "generate_hypotheses": "Generating hypotheses",
        "invoke_tabular_agent": "Delegating to the Tabular Agent",
        "invoke_document_agent": "Delegating to the Document Agent",
        "generate_csv": "Generating a CSV export",
        "generate_markdown_report": "Writing a report",
        "generate_dashboard": "Building a dashboard",
        "get_current_date": "Checking today's date",
        "recall_user_info": "Recalling saved preferences",
        "store_user_info": "Saving a preference",
    }

    @classmethod
    def _translate_event(cls, event) -> dict | None:
        event_type = type(event).__name__
        if event_type == "ToolCallRequestEvent":
            calls = list(event.content)
            names = [c.name for c in calls]
            message = "; ".join(cls._FRIENDLY_TOOL_NAMES.get(n, n) for n in names)
            return {
                "type": "tool_call",
                "message": message,
                "data": {"tools": names},
            }
        if event_type == "ToolCallExecutionEvent":
            names = [res.name for res in event.content]
            message = "; ".join(f"{cls._FRIENDLY_TOOL_NAMES.get(n, n)} - done" for n in names)
            return {
                "type": "tool_result",
                "message": message,
                "data": {"tools": names},
            }
        return None

    def _parse(self, raw: str) -> OrchestratorResult:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("agent did not return valid JSON")
            return OrchestratorResult(
                final_answer=raw,
                confidence="low",
                open_questions=self.tools.state.open_questions,
            )

        return OrchestratorResult(
            final_answer=data.get("final_answer", ""),
            confidence=data.get("confidence", "low"),
            artifact_refs=data.get("artifact_refs", []),
            open_questions=data.get("open_questions", self.tools.state.open_questions),
        )
