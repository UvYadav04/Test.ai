import re

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.document.config import SYSTEM_MESSAGE, get_model_config
from agents.events import make_tool_call_translator
from agents.logger import get_agent_logger, log_event
from llm_provider import LLMProvider, get_settings
from tools.document.document_tools import DocumentTools
from tools.orchestrator.models import DocumentFindings
from vectordb.chroma_store import ChromaVectorStore
from vectordb.reranker import CrossEncoderReranker


class DocumentAgent:
    def __init__(self, assigned_files: list, vector_store=None, reranker=None):
        self.logger = get_agent_logger("document_agent")
        model_config = get_model_config()
        # See orchestrator/agent.py's comment on FALLBACK_LLM_PROVIDER - same reasoning here.
        fallback_provider = get_settings().get("FALLBACK_LLM_PROVIDER", "groq")
        provider = LLMProvider(model_config["provider"], fallback_provider=fallback_provider)
        client = provider.get_client(model_config["model"])

        vector_store = vector_store or ChromaVectorStore()
        if reranker is None:
            reranker = CrossEncoderReranker()

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
                self.tools.broad_scan,
            ],
            system_message=SYSTEM_MESSAGE,
            reflect_on_tool_use=False,
            max_tool_iterations=10,
        )

    async def run(self, objective: str, constraints: dict = None, on_event=None) -> DocumentFindings:
        """`on_event`, if given, is an `async def on_event(event: dict) -> None` -
        forwarded here from OrchestratorTools.invoke_document_agent so this
        agent's OWN tool calls (search_documents, get_chunk, ...) also
        surface on the live activity panel, not just "Assigning an agent"
        with nothing in between until it returns."""
        await self.agent.on_reset(CancellationToken())

        constraints = constraints or {}
        task = (
            f"Objective: {objective}\n"
            f"Assigned file_ids: {self.tools.assigned_file_ids}\n"
            f"Constraints: {constraints}"
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
                if on_event is not None:
                    translated = self._translate_event(event)
                    if translated:
                        await on_event(translated)

        self.logger.info("final reply: %s", final_text)
        table_refs = self._extract_refs(transcript, "table_ref")
        chunk_refs = self._extract_refs(transcript, "chunk_id")
        return DocumentFindings(
            summary=final_text,
            artifact_refs=table_refs,
            source_refs=[ref for ref in chunk_refs if ref in final_text],
        )

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
        "broad_scan": "Scanning documents",
    }

    _translate_event = staticmethod(make_tool_call_translator(_FRIENDLY_TOOL_NAMES))

    @staticmethod
    def _extract_refs(transcript: list, key: str) -> list:
        """Pull real ref values straight out of tool results instead of trusting an LLM to
        transcribe them - table_ref/chunk_id are already exact values a tool returned, so
        re-deriving them via a second model call is both an extra round trip and a chance to
        hallucinate or drop one."""
        text = "\n".join(transcript)
        pattern = rf"['\"]{re.escape(key)}['\"]\s*:\s*['\"]([^'\"]+)['\"]"
        refs = []
        for match in re.findall(pattern, text):
            if match and match not in refs:
                refs.append(match)
        return refs
