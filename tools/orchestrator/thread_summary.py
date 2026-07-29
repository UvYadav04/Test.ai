from agents.orchestrator.config import get_model_config
from llm_provider import LLMProvider, get_settings
from tools.llm_call import ask_llm_async

_SUMMARY_MARKER = "SUMMARY:"
_PREFERENCES_MARKER = "NEW_PREFERENCES:"

TURN_ANALYSIS_PROMPT = """You maintain two things for an ongoing data-analysis conversation:

1. A rolling summary of everything that happened BEFORE the conversation's raw recent-turns
window (only relevant once the chat has grown past that window - see below).
2. A short list of durable user preferences/facts worth remembering across future conversations
(e.g. "prefers bar charts over pie charts", "works in the finance team", "always wants dollar
amounts rounded to 2 decimals") - NOT one-off task details specific to a single question.

{fold_section}

Latest turn (for preference extraction only - this turn is still shown to the agent verbatim
elsewhere, so do NOT fold it into the summary yourself):
User: {query}
Assistant: {response}

Reply in exactly this format, both section headers always present:

{summary_marker}
{summary_instruction}

{preferences_marker}
One durable user fact/preference per line, extracted ONLY from the latest turn above. If there's
nothing new, write the single word None."""

_NO_FOLD_SECTION = (
    "Nothing needs folding into the summary this time - the raw recent-turns window still "
    "covers everything so far. Repeat the previous summary back UNCHANGED, word for word:\n"
    "Previous summary:\n{previous_summary}"
)

_FOLD_SECTION = (
    "Previous summary:\n{previous_summary}\n\n"
    "Fold the following older turn(s) into it - these are about to age out of the raw "
    "recent-turns window, so anything worth keeping must be captured here or it's lost. If a "
    "folded-in turn corrects or contradicts something already in the summary, keep only the "
    "corrected version, don't preserve both:\n{fold_turns}"
)


def _format_fold_turns(turns: list[dict]) -> str:
    return "\n\n".join(
        f"User: {t.get('query', '')}\nAssistant: {t.get('response', '')}" for t in turns
    )


def _parse_response(raw: str, fallback_summary: str) -> tuple[str, list[str]]:
    if _PREFERENCES_MARKER not in raw:
        return raw.replace(_SUMMARY_MARKER, "", 1).strip() or fallback_summary, []

    summary_part, _, preferences_part = raw.partition(_PREFERENCES_MARKER)
    summary = summary_part.replace(_SUMMARY_MARKER, "", 1).strip() or fallback_summary

    preferences = []
    for line in preferences_part.strip().splitlines():
        line = line.strip().lstrip("-*").strip()
        if line and line.lower() != "none":
            preferences.append(line)
    return summary, preferences


async def analyze_turn(
    previous_summary: str, query: str, response: str, turns_to_fold: list[dict] | None = None,
) -> tuple[str, list[str]]:
    model_config = get_model_config()
    fallback_provider = get_settings().get("FALLBACK_LLM_PROVIDER", "groq")
    client = LLMProvider(model_config["provider"], fallback_provider=fallback_provider).get_client(model_config["model"])

    fallback_summary = previous_summary or ""
    previous_summary_display = previous_summary or "(none yet - this is the first turn in this chat)"

    if turns_to_fold:
        fold_section = _FOLD_SECTION.format(
            previous_summary=previous_summary_display, fold_turns=_format_fold_turns(turns_to_fold),
        )
        summary_instruction = "The updated summary, with the turn(s) above folded in. Keep it compact plain language - a few sentences, not a transcript."
    else:
        fold_section = _NO_FOLD_SECTION.format(previous_summary=previous_summary_display)
        summary_instruction = "(repeat the previous summary back unchanged, per the instruction above)"

    prompt = TURN_ANALYSIS_PROMPT.format(
        fold_section=fold_section, query=query, response=response,
        summary_marker=_SUMMARY_MARKER, preferences_marker=_PREFERENCES_MARKER,
        summary_instruction=summary_instruction,
    )
    raw = (await ask_llm_async(client, prompt)).strip()
    return _parse_response(raw, fallback_summary)
