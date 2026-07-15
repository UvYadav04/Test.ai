import json

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.logger import get_agent_logger, log_event
from agents.tabular.config import FORMAT_SYSTEM_MESSAGE, SYSTEM_MESSAGE, get_model_config
from llm_provider import LLMProvider
from tools.orchestrator.models import TabularFindings
from tools.tabular.tabular_tools import TabularTools


class TabularAgent:
    def __init__(self, assigned_files: list, storage=None, workspace_id: str = "default"):
        self.logger = get_agent_logger("tabular_agent")
        self.tools = TabularTools(assigned_files, storage=storage, workspace_id=workspace_id)
        model_config = get_model_config()
        client = LLMProvider(model_config["provider"]).get_client(model_config["model"])

        self.agent = AssistantAgent(
            name="tabular_agent",
            model_client=client,
            tools=[
                self.tools.list_allowed_files,
                self.tools.inspect_schema,
                self.tools.sample_rows,
                self.tools.find_join_candidates,
                self.tools.query_data,
                self.tools.aggregate,
                self.tools.describe_column,
                self.tools.validate_result,
            ],
            system_message=SYSTEM_MESSAGE,
            reflect_on_tool_use=False,
            max_tool_iterations=10,
        )

        self.formatter = AssistantAgent(
            name="tabular_formatter",
            model_client=client,
            system_message=FORMAT_SYSTEM_MESSAGE,
        )

    async def run(self, objective: str, constraints: dict = None) -> TabularFindings:
        await self.agent.on_reset(CancellationToken())
        await self.formatter.on_reset(CancellationToken())

        constraints = constraints or {}
        allowed_files = self.tools.list_allowed_files()
        task = (
            f"Objective: {objective}\n"
            f"Assigned files - use these exact file_id/table_name values, do not guess or "
            f"invent others, and you do not need to call list_allowed_files again unless you "
            f"want to re-check them: {allowed_files}\n"
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

    def _parse(self, raw: str) -> TabularFindings:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning("agent did not return valid JSON")
            return TabularFindings(
                summary=raw,
                findings=[],
                limitations="agent did not return valid JSON",
                confidence="low",
            )

        return TabularFindings(
            summary=data.get("summary", ""),
            findings=data.get("findings", []),
            limitations=data.get("limitations", ""),
            confidence=data.get("confidence", "low"),
            artifact_refs=data.get("artifact_refs", []),
        )
