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
            files="\n".join(self._describe_file(f) for f in files) or "none",
            known_findings="\n".join(str(f) for f in known_findings) or "none",
            max_hypotheses=max_hypotheses,
        )

        client = self.llm_provider.get_client()
        raw = ask_llm(client, prompt)
        data = json.loads(raw)

        hypotheses = [Hypothesis(**h) for h in data.get("hypotheses", [])[:max_hypotheses]]
        return HypothesisResult(hypotheses=hypotheses)

    @staticmethod
    def _describe_file(f) -> str:
        """`context["available_files"]` is a tool-call argument the orchestrator LLM builds
        itself, so items here are ALWAYS plain JSON dicts by the time they arrive - never the
        actual FileCatalogEntry objects a prior list_files call returned (those only look like
        that from the model's side; tool-call arguments round-trip through JSON, which drops
        the class entirely). Attribute access (f.filename) unconditionally crashed here with
        "'dict' object has no attribute 'filename'" on literally every non-empty call. The model
        also isn't guaranteed to reuse list_files' exact field names when it retypes this dict
        (seen in practice: {"file_id": ..., "name": ...} instead of "filename", with "file_type"
        dropped entirely) - so fall back across the field names it's plausible for the model to
        have used, and degrade gracefully (rather than KeyError) if a field is missing outright."""
        if not isinstance(f, dict):
            # Defensive only - shouldn't happen given the JSON round-trip above, but a real
            # FileCatalogEntry (or similar) is handled the same way old code assumed.
            name = getattr(f, "filename", None) or getattr(f, "name", None) or str(f)
            file_type = getattr(f, "file_type", None) or "unknown type"
            return f"- {name} ({file_type})"

        name = f.get("filename") or f.get("name") or f.get("file_id") or "unnamed file"
        file_type = f.get("file_type") or f.get("type") or "unknown type"
        return f"- {name} ({file_type})"
