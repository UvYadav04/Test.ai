import json
import uuid

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.logger import get_agent_logger, log_event
from agents.orchestrator.config import FORMAT_SYSTEM_MESSAGE, SYSTEM_MESSAGE, get_model_config
from llm_provider import LLMProvider
from tools.orchestrator.models import InvestigationState, OrchestratorResult
from tools.orchestrator.orchestrator_tools import OrchestratorTools


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

        self.formatter = AssistantAgent(
            name="orchestrator_formatter",
            model_client=client,
            system_message=FORMAT_SYSTEM_MESSAGE,
        )

    async def run(self, objective: str, workspace_id: str = "default", constraints: dict = None) -> OrchestratorResult:
        await self.agent.on_reset(CancellationToken())
        await self.formatter.on_reset(CancellationToken())

        constraints = constraints or {}
        self.tools.workspace_id = workspace_id
        self.tools.state = InvestigationState(
            session_id=uuid.uuid4().hex[:12],
            objective=objective,
            constraints=constraints,
        )

        task = f"Objective: {objective}\nWorkspace: {workspace_id}\nConstraints: {constraints}"
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
            f"Investigation State:\n{self.tools.state.summary()}\n"
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
