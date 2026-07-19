import re

from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken

from agents.events import make_tool_call_translator
from agents.logger import get_agent_logger, log_event
from agents.tabular.config import SYSTEM_MESSAGE, get_model_config
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
                self.tools.run_python,
            ],
            system_message=SYSTEM_MESSAGE,
            reflect_on_tool_use=False,
            max_tool_iterations=10,
        )

    async def run(self, objective: str, constraints: dict = None, on_event=None) -> TabularFindings:
        """`on_event`, if given, is an `async def on_event(event: dict) -> None` -
        forwarded here from OrchestratorTools.invoke_tabular_agent so this
        agent's OWN tool calls (run_python, list_allowed_files) also surface
        on the live activity panel, not just "Assigning an agent" with
        nothing in between until it returns."""
        await self.agent.on_reset(CancellationToken())

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
        return TabularFindings(
            summary=final_text,
            artifact_refs=self._extract_refs(transcript, "output_ref"),
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
        "list_allowed_files": "Listing files",
        "run_python": "Executing a Python script",
    }

    _translate_event = staticmethod(make_tool_call_translator(_FRIENDLY_TOOL_NAMES))

    @staticmethod
    def _extract_refs(transcript: list, key: str) -> list:
        """Pull real output_ref paths straight out of tool results (e.g. run_python's
        save() entries) instead of trusting an LLM to transcribe them - the sandbox already
        returns the exact path, so re-deriving it from a second model call is both an extra
        round trip and a chance to hallucinate or drop it."""
        text = "\n".join(transcript)
        pattern = rf"['\"]{re.escape(key)}['\"]\s*:\s*['\"]([^'\"]+)['\"]"
        refs = []
        for match in re.findall(pattern, text):
            if match and match not in refs:
                refs.append(match)
        return refs
