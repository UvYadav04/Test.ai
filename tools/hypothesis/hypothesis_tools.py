import json

from llm_provider import LLMProvider
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
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider or LLMProvider()

    def generate_hypotheses(self, objective: str, context: dict, max_hypotheses: int = 5) -> HypothesisResult:
        files = context.get("available_files", [])
        known_findings = context.get("known_findings") or []

        prompt = PROMPT_TEMPLATE.format(
            objective=objective,
            files="\n".join(f"- {f.filename} ({f.file_type})" for f in files) or "none",
            known_findings="\n".join(str(f) for f in known_findings) or "none",
            max_hypotheses=max_hypotheses,
        )

        client = self.llm_provider.get_client()
        raw = ask_llm(client, prompt)
        data = json.loads(raw)

        hypotheses = [Hypothesis(**h) for h in data.get("hypotheses", [])[:max_hypotheses]]
        return HypothesisResult(hypotheses=hypotheses)
