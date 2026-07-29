import json

from llm_provider import LLMProvider
from tools.hypothesis.config import get_model_config
from tools.hypothesis.models import Hypothesis, HypothesisResult
from tools.llm_call import ask_llm

PROMPT_TEMPLATE = """You are helping investigate a data question.

Objective: {objective}

Available files:
{files}

Known findings so far:
{known_findings}

Generate up to {max_hypotheses} candidate hypotheses that could explain or answer the objective.
Return ONLY valid JSON in this exact shape, nothing else:
{{"hypotheses": [{{"statement": "...", "suggested_investigation": "...", "suggested_agent": "tabular|document|both", "priority": 1}}]}}
Order hypotheses by priority, 1 = highest.
"""


class HypothesisTools:
    def __init__(self, llm_provider=None, model: str = None):
        if llm_provider is not None:
            self.llm_provider = llm_provider
            self.model = model
        else:
            model_config = get_model_config()
            self.llm_provider = LLMProvider(model_config["provider"])
            self.model = model or model_config["model"]

    def generate_hypotheses(self, objective: str, context: dict, max_hypotheses: int = 5) -> HypothesisResult:
        files = context.get("available_files", [])
        known_findings = context.get("known_findings") or []

        prompt = PROMPT_TEMPLATE.format(
            objective=objective,
            files="\n".join(self._describe_file(f) for f in files) or "none",
            known_findings="\n".join(str(f) for f in known_findings) or "none",
            max_hypotheses=max_hypotheses,
        )

        client = self.llm_provider.get_client(self.model)
        raw = ask_llm(client, prompt)
        data = json.loads(raw)

        hypotheses = [Hypothesis(**h) for h in data.get("hypotheses", [])[:max_hypotheses]]
        return HypothesisResult(hypotheses=hypotheses)

    @staticmethod
    def _describe_file(f) -> str:
        if not isinstance(f, dict):
            name = getattr(f, "filename", None) or getattr(f, "name", None) or str(f)
            file_type = getattr(f, "file_type", None) or "unknown type"
            return f"- {name} ({file_type})"

        name = f.get("filename") or f.get("name") or f.get("file_id") or "unnamed file"
        file_type = f.get("file_type") or f.get("type") or "unknown type"
        return f"- {name} ({file_type})"
