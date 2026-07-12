import json

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.document.config import FORMAT_SYSTEM_MESSAGE, SYSTEM_MESSAGE, get_model_config
from agents.logger import get_agent_logger, log_event
from llm_provider import LLMProvider
from tools.document.document_tools import DocumentTools
from tools.orchestrator.models import DocumentFindings
from vectordb.chroma_store import ChromaVectorStore
from vectordb.reranker import CrossEncoderReranker


class DocumentAgent:
    def __init__(self, assigned_files: list, vector_store=None, reranker=None):
        self.logger = get_agent_logger("document_agent")
        model_config = get_model_config()
        provider = LLMProvider(model_config["provider"])
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

        self.formatter = AssistantAgent(
            name="document_formatter",
            model_client=client,
            system_message=FORMAT_SYSTEM_MESSAGE,
        )

    async def run(self, objective: str, constraints: dict = None) -> DocumentFindings:
        await self.agent.on_reset(CancellationToken())
        await self.formatter.on_reset(CancellationToken())

        constraints = constraints or {}
        task = (
            f"Objective: {objective}\n"
            f"Assigned file_ids: {self.tools.assigned_file_ids}\n"
            f"Constraints: {constraints}"
        )
        self.logger.info("objective sent to agent: %s", task)

        transcript = []
        async for event in self.agent.run_stream(task=task):
            if not hasattr(event, "messages"):
                log_event(self.logger, event)
                line = self._transcript_line(event)
                if line:
                    transcript.append(line)

        format_task = (
            f"Objective: {objective}\n"
            "Tool activity:\n" + "\n".join(transcript) +
            "\nRespond with ONLY the final JSON now."
        )
        format_result = await self.formatter.run(task=format_task)
        raw = format_result.messages[-1].content
        self.logger.info("final reply: %s", raw)
        return self._parse(raw)

    @staticmethod
    def _transcript_line(event) -> str:
        event_type = type(event).__name__
        if event_type == "ToolCallRequestEvent":
            return "\n".join(f"CALL {call.name}({call.arguments})" for call in event.content)
        if event_type == "ToolCallExecutionEvent":
            return "\n".join(f"RESULT {res.name} -> {res.content}" for res in event.content)
        return ""

    def _parse(self, raw: str) -> DocumentFindings:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("agent did not return valid JSON")
            return DocumentFindings(
                summary=raw,
                findings=[],
                limitations="agent did not return valid JSON",
                confidence="low",
            )

        return DocumentFindings(
            summary=data.get("summary", ""),
            findings=data.get("findings", []),
            limitations=data.get("limitations", ""),
            confidence=data.get("confidence", "low"),
            artifact_refs=data.get("artifact_refs", []),
            source_refs=data.get("source_refs", []),
        )
