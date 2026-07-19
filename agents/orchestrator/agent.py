import re
import uuid
from datetime import datetime, timezone

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.events import make_tool_call_translator
from agents.logger import get_agent_logger, log_event
from agents.orchestrator.config import SYSTEM_MESSAGE, get_model_config
from llm_provider import LLMProvider
from tools.orchestrator.models import InvestigationState, OrchestratorResult
from tools.orchestrator.orchestrator_tools import OrchestratorTools

_DELIVERABLE_TOOLS = {"generate_csv", "generate_markdown_report", "generate_dashboard"}


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

    async def run(
        self,
        objective: str,
        workspace_id: str = "default",
        constraints: dict = None,
        thread_context: dict = None,
        on_event=None,
        cancel_check=None,
    ) -> OrchestratorResult:
        """`on_event`, if given, is an `async def on_event(event: dict) -> None`
        called once per meaningful step (tool call requested/executed) - see
        `_translate_event` for the event shapes. `cancel_check`, if given, is
        an `async def cancel_check() -> bool` polled between steps (never
        mid-tool-call/mid-LLM-call); returning True stops the loop cleanly
        and raises InvestigationCancelled instead of returning a result.

        `thread_context`, if given, is a dict with the calling chat's
        {summary, recent_turns, files_used, files_created} - see
        shared/models/chat.py and _thread_context_brief. This agent instance
        is built fresh per job and never remembers anything between calls
        itself (on_reset() below is explicit about that), so this is the
        ONLY way an earlier message in the same chat reaches this run -
        worker_service.tasks.investigation reads it off the Chat doc before
        calling this and writes the updated version back after.

        The orchestrator's own final reply (a plain-language TextMessage, per
        SYSTEM_MESSAGE) is used as-is for `final_answer` - there's no second
        LLM call reformatting it, so nothing can drift away from what the
        agent actually concluded."""
        await self.agent.on_reset(CancellationToken())

        constraints = constraints or {}
        self.tools.workspace_id = workspace_id
        self.tools.state = InvestigationState(
            session_id=uuid.uuid4().hex[:12],
            objective=objective,
            constraints=constraints,
        )
        # Handed to invoke_tabular_agent/invoke_document_agent so the
        # delegated sub-agent's own tool calls (run_python, search_documents,
        # ...) stream as events too, not just the orchestrator's - see
        # TabularAgent.run/DocumentAgent.run's own `on_event` param.
        self.tools.on_event = on_event

        task = (
            f"Objective: {objective}\n"
            f"Workspace: {workspace_id}\n"
            f"Constraints: {constraints}\n\n"
            f"{self._thread_context_brief(thread_context)}\n\n"
            f"{self._context_brief()}"
        )
        self.logger.info("objective sent to agent: %s", task)

        transcript = []
        final_text = ""
        stream = self.agent.run_stream(task=task)
        try:
            async for event in stream:
                if not hasattr(event, "messages"):
                    log_event(self.logger, event)
                    line = self._transcript_line(event)
                    if line:
                        transcript.append(line)
                    if type(event).__name__ == "TextMessage" and getattr(event, "source", None) == self.agent.name:
                        final_text = event.content
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

        self.logger.info("final reply: %s", final_text)
        return OrchestratorResult(
            final_answer=final_text,
            artifact_refs=self._collect_artifact_refs(transcript),
            open_questions=self.tools.state.open_questions,
            # Dedup while preserving first-seen order - dict.fromkeys is the
            # idiomatic way to do that without reaching for a separate set +
            # list. worker_service.tasks.investigation merges this into the
            # Chat's files_used after the run, for the NEXT investigation in
            # this chat to see via thread_context.
            files_used=list(dict.fromkeys(self.tools.state.selected_files)),
        )

    def _thread_context_brief(self, thread_context: dict | None) -> str:
        """Continuity from earlier turns in THIS chat - distinct from
        _context_brief's workspace-wide file catalog. Comes from
        Chat.summary/recent_turns/files_used/files_created
        (shared/models/chat.py), refreshed after every completed
        investigation in this chat by
        worker_service.tasks.investigation._update_chat_continuity - this
        method only ever formats whatever it's handed, it never reads or
        writes anything itself."""
        if not thread_context:
            return "This is the first message in this chat - no earlier context."

        lines = []

        summary = thread_context.get("summary")
        if summary:
            lines.append(f"Summary of this chat so far: {summary}")

        recent_turns = thread_context.get("recent_turns") or []
        if recent_turns:
            lines.append("Most recent turns in this chat (oldest first) - use these to resolve "
                         "references like \"that file\", \"the same but by region\", or a "
                         "correction to what you said before:")
            for turn in recent_turns:
                lines.append(f"- User: {turn.get('query', '')}")
                lines.append(f"  You answered: {turn.get('response', '')}")

        files_used = thread_context.get("files_used") or []
        if files_used:
            lines.append(f"file_ids already queried earlier in this chat: {files_used}")

        files_created = thread_context.get("files_created") or []
        if files_created:
            lines.append(f"Artifacts already produced earlier in this chat: {files_created}")

        return "\n".join(lines) if lines else "This is the first message in this chat - no earlier context."

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

    # Human-readable labels for the homescreen "live activity" panel (see
    # project_documentation_and_claude_code_guide.md Section 6) - plain-
    # language status, not raw tool logs, and never the internal name of a
    # delegated agent (invoke_tabular_agent/invoke_document_agent both just
    # read as "Assigning an agent" - the user sees what's happening, not
    # which specialist is doing it). See agents/events.py for why there's no
    # matching "done" event for any of these.
    _FRIENDLY_TOOL_NAMES = {
        "list_files": "Listing files",
        "search_files": "Searching files",
        "get_file_details": "Getting file metadata",
        "list_tables": "Listing tables",
        "list_file_formats": "Checking file types",
        "generate_hypotheses": "Generating hypotheses",
        "invoke_tabular_agent": "Assigning an agent",
        "invoke_document_agent": "Assigning an agent",
        "generate_csv": "Exporting a CSV",
        "generate_markdown_report": "Writing a report",
        "generate_dashboard": "Building a dashboard",
        "get_current_date": "Checking today's date",
        "recall_user_info": "Recalling saved preferences",
        "store_user_info": "Saving a preference",
    }

    _translate_event = staticmethod(make_tool_call_translator(_FRIENDLY_TOOL_NAMES))

    def _collect_artifact_refs(self, transcript: list) -> list:
        """Real artifact paths only - never something an LLM transcribed and could get wrong.
        Two sources: (1) whatever the delegated Tabular/Document agents already reported on
        their own TabularFindings/DocumentFindings.artifact_refs (e.g. a table_ref surfaced
        from a PDF), read straight off InvestigationState.completed_tasks; (2) any file path a
        generate_csv/generate_markdown_report/generate_dashboard call actually returned, parsed
        out of its RESULT line in the tool-call transcript."""
        refs = []
        for event in self.tools.state.completed_tasks:
            for ref in getattr(event.result, "artifact_refs", None) or []:
                if ref not in refs:
                    refs.append(ref)

        pattern = re.compile(r"^RESULT (\w+) -> (.+)$")
        for line in transcript:
            for sub_line in line.split("\n"):
                match = pattern.match(sub_line)
                if not match or match.group(1) not in _DELIVERABLE_TOOLS:
                    continue
                ref = match.group(2).strip()
                if ref and ref not in refs:
                    refs.append(ref)

        return refs
