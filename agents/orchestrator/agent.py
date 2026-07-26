import inspect
import re
import time
import uuid
from datetime import datetime, timezone
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import UnboundedChatCompletionContext

from agents.events import make_tool_call_translator
from agents.logger import get_agent_logger, log_event
from agents.orchestrator import capabilities
from agents.orchestrator.config import SYSTEM_MESSAGE, get_model_config
from agents.timing import ToolCallTimer
from llm_provider import LLMProvider, get_settings
from tools.orchestrator.models import InvestigationState, OrchestratorResult
from tools.orchestrator.orchestrator_tools import OrchestratorTools

_DELIVERABLE_TOOLS = {"generate_csv", "generate_markdown_report", "generate_dashboard"}

_AGENT_NAME = "orchestrator_agent"

# Upper bound on outer (= one LLM call each, see run()'s loop) iterations - same budget the
# single AssistantAgent used to get via max_tool_iterations=25 before this file switched to
# reconstructing a fresh AssistantAgent per iteration (see run()'s docstring for why).
_MAX_OUTER_ITERATIONS = 25


class _CapabilityHolder:
    """Tiny mutable box holding the most recent `next_capabilities` value any wrapped tool call
    received this run - see `_wrap_with_next_capabilities` below. One instance per
    OrchestratorAgent.run() call, reset before each outer-loop iteration so a turn that makes no
    tool call (a plain-text final answer) can never see a stale value from an earlier turn."""

    __slots__ = ("value",)

    def __init__(self):
        self.value: list[str] = []


def _wrap_with_next_capabilities(func, holder: "_CapabilityHolder"):
    """Generically adds one extra parameter, `next_capabilities: list[str] = []`, to ANY
    orchestrator tool callable's exposed schema - this is applied uniformly to every tool (core
    and capability-gated alike), never special-cased per tool name, so registering a brand new
    capability never requires touching this function.

    Why the extra param lives on every tool's own call instead of being a separate
    `set_next_capabilities` tool: every LLM provider here is configured with
    parallel_tool_calls=False (see llm_provider/providers/*.py, and agents/timing.py's note on
    why), so the model can only make ONE tool call per turn - it has no way to call e.g.
    invoke_tabular_agent AND a hypothetical separate capability-request tool in the same turn.
    Piggybacking `next_capabilities` onto whatever tool the model was already going to call is
    the only way to get both "the real action" and "what's needed next" out of a single model
    round trip, which is what keeps this mechanism from adding any extra LLM calls or prompt
    round-trips beyond what the orchestrator already made.

    The real tool (`func`) is never modified and is called with its original arguments only -
    `next_capabilities` is popped off before the call. `holder.value` is overwritten on every
    call (default `[]` if the model didn't supply one), and is read by run()'s outer loop right
    after the call returns to decide which capabilities to expose on the next iteration.
    """
    orig_sig = inspect.signature(func)
    next_capabilities_param = inspect.Parameter(
        "next_capabilities",
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=[],
        annotation=List[str],
    )
    new_sig = orig_sig.replace(parameters=[*orig_sig.parameters.values(), next_capabilities_param])

    if inspect.iscoroutinefunction(func):
        async def wrapper(**kwargs):
            holder.value = list(kwargs.pop("next_capabilities", None) or [])
            return await func(**kwargs)
    else:
        def wrapper(**kwargs):
            holder.value = list(kwargs.pop("next_capabilities", None) or [])
            return func(**kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__signature__ = new_sig
    wrapper.__annotations__ = {**getattr(func, "__annotations__", {}), "next_capabilities": List[str]}
    base_doc = (func.__doc__ or "").rstrip()
    wrapper.__doc__ = (
        f"{base_doc}\n\n"
        "next_capabilities (optional): declare which additional capability-gated tools you'll "
        "need for your VERY NEXT call, so they're added to your toolset without bloating every "
        "turn's prompt - pass their names as a list, e.g. [\"dashboard\"]. Registered "
        f"capabilities:\n{capabilities.capability_catalog_text()}\n"
        "Leave this empty (or omit it) if your next step doesn't need anything beyond your "
        "current tools - capabilities requested here stay available for one step only, ask "
        "again if you still need them after that."
    )
    return wrapper


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
        reports_dir: str = "data/reports", investigation_id: str = "default", sandbox_manager=None,
    ):
        self.logger = get_agent_logger("orchestrator_agent")
        model_config = get_model_config()
        # FALLBACK_LLM_PROVIDER (defaults to "groq" for backward compat) - if it ends up equal
        # to model_config["provider"] (e.g. ORCHESTRATOR_PROVIDER=groq with the default fallback
        # left as-is), LLMProvider.get_client() detects that and skips the no-op fallback wrap
        # entirely, so a primary-provider outage/rate-limit isn't silently "protected" by nothing.
        # Set FALLBACK_LLM_PROVIDER to a provider genuinely different from ORCHESTRATOR_PROVIDER
        # for the fallback to do anything.
        fallback_provider = get_settings().get("FALLBACK_LLM_PROVIDER", "groq")
        client = LLMProvider(model_config["provider"], fallback_provider=fallback_provider).get_client(model_config["model"])



        self.tools = OrchestratorTools(
            catalog, state=None, vector_store=vector_store, reranker=reranker, memory=memory, storage=storage,
            reports_dir=reports_dir, investigation_id=investigation_id, sandbox_manager=sandbox_manager,
        )
        self.model_client = client

        # Every tool the orchestrator could ever call, each wrapped once (not re-wrapped per
        # run/iteration - wrapping only touches the callable's exposed schema/doc, it doesn't
        # capture any per-run state itself) with the generic `next_capabilities` parameter. A
        # single shared _CapabilityHolder is threaded through per run() call (see there) so
        # whichever tool the model actually calls each turn writes into the same place.
        self._capability_holder = _CapabilityHolder()
        self._wrapped_tools = {
            name: _wrap_with_next_capabilities(getattr(self.tools, name), self._capability_holder)
            for name in capabilities.all_tool_names()
        }

    async def run(
        self,
        objective: str,
        workspace_id: str = "default",
        constraints: dict = None,
        thread_context: dict = None,
        on_event=None,
        cancel_check=None,
    ) -> OrchestratorResult:


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

        run_start = time.perf_counter()
        tool_timer = ToolCallTimer(self.logger)
        transcript = []
        final_text = ""

        model_context = UnboundedChatCompletionContext()
        active_capabilities: list[str] = []
        next_task = task

        for outer_iteration in range(_MAX_OUTER_ITERATIONS):
            tool_names = capabilities.CORE_TOOLS + capabilities.tools_for_capabilities(active_capabilities)
            iteration_agent = AssistantAgent(
                name=_AGENT_NAME,
                model_client=self.model_client,
                tools=[self._wrapped_tools[name] for name in tool_names],
                model_context=model_context,
                system_message=SYSTEM_MESSAGE,
                reflect_on_tool_use=False,
                max_tool_iterations=1,
            )
            # Reset before every iteration (not just the first) so a turn that ends without any
            # tool call - i.e. the model's final plain-text answer - can never pick up a stale
            # value left over from an earlier turn.
            self._capability_holder.value = []
            ended_in_final_answer = False

            self.logger.info(
                "orchestrator outer iteration %d: exposing %d tools (%s)",
                outer_iteration, len(tool_names), ", ".join(tool_names),
            )
            stream = iteration_agent.run_stream(task=next_task)
            # Only the very first outer iteration sends the objective as a new user message -
            # every iteration after that continues the same model_context (see docstring above).
            next_task = None
            try:
                async for event in stream:
                    if not hasattr(event, "messages"):
                        log_event(self.logger, event)
                        tool_timer.record(event)
                        line = self._transcript_line(event)
                        if line:
                            transcript.append(line)
                        if type(event).__name__ == "TextMessage" and getattr(event, "source", None) == _AGENT_NAME:
                            final_text = event.content
                            ended_in_final_answer = True
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

            if ended_in_final_answer:
                # No tool call this turn - matches autogen's own stopping rule ("if the model
                # returns no tool call... this ends the tool call iteration loop regardless of
                # the max_tool_iterations setting"), just enforced by this outer loop instead.
                break

            # The model's next_capabilities (from whichever tool it called this turn, captured
            # by _wrap_with_next_capabilities) decide what's exposed on the NEXT iteration only -
            # not accumulated - so the prompt stays as small as the model says it needs it to be.
            active_capabilities = list(self._capability_holder.value)
        else:
            self.logger.warning(
                "orchestrator hit _MAX_OUTER_ITERATIONS (%d) without a final text answer",
                _MAX_OUTER_ITERATIONS,
            )

        self.logger.info("orchestrator agent run took %.3fs", time.perf_counter() - run_start)
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

        # .browsable(), not .all(): same visibility rules as list_files/search_files (see
        # FileCatalog.is_browsable) - an xlsx workbook's own file_id never appears here (it has
        # no queryable data of its own, only its sheets do - see list_tables), and a PDF's
        # per-page tables are never pre-enumerated here (a dense PDF can have dozens; this brief
        # is rebuilt fresh into every single task message, so listing them all would blow up
        # context turn after turn - they're meant to surface lazily via a Document Agent's own
        # table_ref instead).
        entries = self.tools.catalog.browsable()
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
        "invoke_document_processor": "Assigning an agent",
        "generate_csv": "Exporting a CSV",
        "generate_markdown_report": "Writing a report",
        "generate_dashboard": "Building a real-time dashboard",
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
