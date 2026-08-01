import inspect
import time
import uuid
from datetime import datetime, timezone
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_core.model_context import UnboundedChatCompletionContext

from agents.events import make_tool_event_translator
from agents.final_answer import split_follow_up_questions
from agents.logger import get_agent_logger, log_event
from agents.orchestrator import capabilities
from agents.orchestrator.config import SYSTEM_MESSAGE, get_model_config
from agents.thread_context import thread_context_brief
from agents.timing import ToolCallTimer
from llm_provider import LLMProvider, get_settings
from tools.orchestrator.models import FinalResultCollector, InvestigationState, OrchestratorResult
from tools.orchestrator.orchestrator_tools import OrchestratorTools

_AGENT_NAME = "orchestrator_agent"

_MAX_OUTER_ITERATIONS = 25


class _CapabilityHolder:
    __slots__ = ("value",)

    def __init__(self):
        self.value: list[str] = []


def _wrap_with_next_capabilities(func, holder: "_CapabilityHolder"):
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
    def __init__(self, state: InvestigationState):
        super().__init__("investigation cancelled")
        self.state = state


class OrchestratorAgent:
    def __init__(
        self, catalog, vector_store=None, reranker=None, memory=None, storage=None,
        reports_dir: str = "data/reports", chat_id: str = "default", sandbox_manager=None,
        result_collector: FinalResultCollector = None, chart_capacity_checker=None,
    ):
        self.logger = get_agent_logger("orchestrator_agent")
        model_config = get_model_config()
        fallback_provider = get_settings().get("FALLBACK_LLM_PROVIDER", "groq")
        client = LLMProvider(model_config["provider"], fallback_provider=fallback_provider).get_client(model_config["model"])

        self.tools = OrchestratorTools(
            catalog, state=None, vector_store=vector_store, reranker=reranker, memory=memory, storage=storage,
            reports_dir=reports_dir, chat_id=chat_id, sandbox_manager=sandbox_manager,
            result_collector=result_collector or FinalResultCollector(),
            chart_capacity_checker=chart_capacity_checker,
        )
        self.model_client = client

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
        self.tools.on_event = on_event

        task = (
            f"Objective: {objective}\n"
            f"Workspace: {workspace_id}\n"
            f"Constraints: {constraints}\n\n"
            f"{thread_context_brief(thread_context)}\n\n"
            f"{self._context_brief()}"
        )
        self.logger.info("objective sent to agent: %s", task)

        run_start = time.perf_counter()
        tool_timer = ToolCallTimer(self.logger)
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
            self._capability_holder.value = []
            ended_in_final_answer = False

            self.logger.info(
                "orchestrator outer iteration %d: exposing %d tools (%s)",
                outer_iteration, len(tool_names), ", ".join(tool_names),
            )
            stream = iteration_agent.run_stream(task=next_task)
            next_task = None
            try:
                async for event in stream:
                    if not hasattr(event, "messages"):
                        log_event(self.logger, event)
                        tool_timer.record(event)
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
                break

            active_capabilities = list(self._capability_holder.value)
        else:
            self.logger.warning(
                "orchestrator hit _MAX_OUTER_ITERATIONS (%d) without a final text answer",
                _MAX_OUTER_ITERATIONS,
            )

        self.logger.info("orchestrator agent run took %.3fs", time.perf_counter() - run_start)
        collector = self.tools.result_collector
        answer, follow_up_questions = split_follow_up_questions(final_text)
        return OrchestratorResult(
            final_answer=answer,
            chart_paths=collector.chart_paths,
            artifacts=collector.artifacts,
            files_used=collector.files_used,
            open_questions=self.tools.state.open_questions,
            follow_up_questions=follow_up_questions,
        )

    def _context_brief(self, max_files: int = 40, max_columns: int = 25) -> str:
        now = datetime.now(timezone.utc)
        lines = [f"Today's date: {now.date().isoformat()} ({now.strftime('%A')}, UTC)."]

        user_info = self.tools.memory.recall_all()
        if user_info:
            lines.append("Known standing user preferences/facts (from recall_user_info):")
            lines.extend(f"- {fact}" for fact in user_info)
        else:
            lines.append("No standing user preferences/facts saved yet.")

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
        "generate_report": "Writing a report",
        "generate_dashboard": "Building a real-time dashboard",
        "get_current_date": "Checking today's date",
        "recall_user_info": "Recalling saved preferences",
    }

    _translate_event = staticmethod(make_tool_event_translator(_FRIENDLY_TOOL_NAMES))
