import re
import time

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.document.config import get_model_config, get_system_message
from agents.events import make_tool_event_translator
from agents.final_answer import split_follow_up_questions
from agents.logger import get_agent_logger, log_event
from agents.thread_context import thread_context_brief
from agents.timing import ToolCallTimer
from llm_provider import LLMProvider, get_settings
from tools.document.document_tools import DocumentTools
from tools.orchestrator.models import DocumentFindings, InvestigationCancelled
from vectordb.chroma_store import ChromaVectorStore
from vectordb.reranker import DeepInfraReranker


class DocumentAgent:
    def __init__(self, assigned_files: list, vector_store=None, reranker=None, direct_route: bool = False):
        self.logger = get_agent_logger("document_agent")
        model_config = get_model_config()
        fallback_provider = get_settings().get("FALLBACK_LLM_PROVIDER", "groq")
        provider = LLMProvider(model_config["provider"], fallback_provider=fallback_provider)
        client = provider.get_client(model_config["model"])

        vector_store = vector_store or ChromaVectorStore()
        if reranker is None:
            reranker = DeepInfraReranker()

        self.tools = DocumentTools(assigned_files, vector_store, reranker=reranker, llm_provider=provider)

        self.agent = AssistantAgent(
            name="document_agent",
            model_client=client,
            tools=[
                self.tools.get_file_overview,
                self.tools.expand_query,
                self.tools.search_documents,
                self.tools.search_within_file,
                self.tools.get_chunk,
                self.tools.get_surrounding_chunks,
                self.tools.list_file_sections,
                self.tools.compare_documents,
                self.tools.search_for_contradictions,
                self.tools.verify_chunk_supports_claim,
                self.tools.list_tables,
                self.tools.search_tables,
                self.tools.get_table,
            ],
            system_message=get_system_message(direct_route),
            reflect_on_tool_use=False,
            max_tool_iterations=10,
        )

    async def run(
        self, objective: str, constraints: dict = None, on_event=None, metadata_brief: str = None,
        thread_context: dict = None, cancel_check=None,
    ) -> DocumentFindings:

        await self.agent.on_reset(CancellationToken())

        constraints = constraints or {}
        task = (
            f"Objective: {objective}\n"
            f"Assigned file_ids: {self.tools.assigned_file_ids}\n"
            f"Constraints: {constraints}\n\n"
            f"{thread_context_brief(thread_context)}\n\n"
            f"{self._metadata_section(metadata_brief)}"
        )
        self.logger.info("objective sent to agent: %s", task)

        run_start = time.perf_counter()
        tool_timer = ToolCallTimer(self.logger)
        transcript = []
        final_text = ""
        # Checked after every streamed event, same as OrchestratorAgent.run() - without this, a
        # cancel request made while this agent is mid-way through several search/verify tool
        # calls (max_tool_iterations=10) had no effect until the whole nested run finished on its
        # own, since the orchestrator's own cancel_check only fires again once
        # invoke_document_agent's single tool call returns.
        stream = self.agent.run_stream(task=task)
        try:
            async for event in stream:
                if not hasattr(event, "messages"):
                    log_event(self.logger, event)
                    tool_timer.record(event)
                    line = self._transcript_line(event)
                    if line:
                        transcript.append(line)
                    if type(event).__name__ == "TextMessage" and event.source == self.agent.name:
                        final_text = event.content
                    if on_event is not None:
                        translated = self._translate_event(event)
                        if translated:
                            await on_event(translated)

                if cancel_check is not None and await cancel_check():
                    await stream.aclose()
                    if on_event is not None:
                        await on_event({"type": "cancelled", "message": "Investigation cancelled."})
                    raise InvestigationCancelled()
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

        self.logger.info("final reply: %s", final_text)
        self.logger.info("document agent run took %.3fs", time.perf_counter() - run_start)
        table_refs = self._extract_refs(transcript, "table_ref")
        chunk_refs = self._extract_refs(transcript, "chunk_id")
        summary, follow_up_questions = split_follow_up_questions(final_text)
        return DocumentFindings(
            summary=summary,
            artifact_refs=table_refs,
            source_refs=[ref for ref in chunk_refs if ref in summary],
            follow_up_questions=follow_up_questions,
        )

    @staticmethod
    def _metadata_section(metadata_brief: str | None) -> str:
        if not metadata_brief:
            return ""
        return (
            "Document metadata (already retrieved deterministically - do NOT call "
            "get_file_overview or list_file_sections/list_tables again just to re-learn this, "
            "use your first real tool call for the actual objective instead):\n"
            f"{metadata_brief}"
        )

    @staticmethod
    def _transcript_line(event) -> str:
        event_type = type(event).__name__
        if event_type == "ToolCallRequestEvent":
            return "\n".join(f"CALL {call.name}({call.arguments})" for call in event.content)
        if event_type == "ToolCallExecutionEvent":
            return "\n".join(f"RESULT {res.name} -> {res.content}" for res in event.content)
        return ""

    _FRIENDLY_TOOL_NAMES = {
        "get_file_overview": "Reviewing file details",
        "expand_query": "Refining the search",
        "search_documents": "Searching documents",
        "search_within_file": "Searching within a file",
        "get_chunk": "Reading document content",
        "get_surrounding_chunks": "Reading surrounding context",
        "list_file_sections": "Listing document sections",
        "compare_documents": "Comparing documents",
        "search_for_contradictions": "Checking for contradictions",
        "verify_chunk_supports_claim": "Verifying a finding",
        "list_tables": "Listing tables",
        "search_tables": "Searching tables",
        "get_table": "Getting table metadata",
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
