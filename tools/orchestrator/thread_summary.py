"""Maintains Chat.summary and long-term user preferences - both regenerated from one small LLM
call per completed investigation (see worker_service.tasks.investigation.update_chat_memory, an
arq job enqueued AFTER the investigation's own "completed" event/Message are already persisted
and broadcast, so this call's latency is never on the user-facing critical path).

Two jobs share this one call rather than two separate ones:

1. Rolling chat summary - only needed once a chat has more turns than fit in the raw
   recent_turns window shown verbatim to the orchestrator (see
   OrchestratorAgent._thread_context_brief and RECENT_TURNS_LIMIT in investigation.py).
   Summarizing every turn regardless of chat length (the old behavior) meant the orchestrator's
   prompt carried the same turn twice - once in the summary, once verbatim in recent_turns - for
   the first RECENT_TURNS_LIMIT turns of every single chat. Callers only pass turns_to_fold once
   a turn is actually about to age out of that window; otherwise the previous summary is
   returned unchanged with no LLM involvement at all for that half of the call... except this is
   one combined call (see below), so the model is simply told there's nothing to fold this time.

2. Durable user preferences/facts (e.g. "prefers bar charts over pie charts") - extracted from
   EVERY completed turn, independent of chat length. This replaces the orchestrator's old
   store_user_info tool call: previously the orchestrator had to notice something worth
   remembering mid-run and spend a whole extra synchronous LLM round trip (tool call -> tool
   result -> continue) calling it. Now the same extraction happens for free as a side effect of
   this already-scheduled, already-async summary call - zero added latency during the
   investigation itself.

Deliberately its own tiny LLM call rather than something folded into the orchestrator's own run:
it only ever sees {previous_summary, turns to fold, latest user message, latest final answer} -
no tool schemas, no file catalog - so it stays small and fast regardless of how large the
orchestrator's own prompt grows, and a failure here (see call site) never has to fail the
investigation itself.
"""
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
    """Defensive parsing - if the model doesn't follow the exact format, fall back to treating
    the whole reply as the summary and reporting no new preferences, rather than raising and
    losing the turn's continuity entirely (see call site's own try/except for the outer
    failure/retry story)."""
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
    """Returns (new_summary, new_preference_lines). new_summary equals previous_summary
    (functionally - modulo the model faithfully echoing it back, see _NO_FOLD_SECTION) whenever
    turns_to_fold is empty; new_preference_lines is extracted from (query, response) every call,
    regardless of turns_to_fold."""
    model_config = get_model_config()
    # See orchestrator/agent.py's comment on FALLBACK_LLM_PROVIDER - same reasoning here.
    fallback_provider = get_settings().get("FALLBACK_LLM_PROVIDER", "groq")
    client = LLMProvider(model_config["provider"], fallback_provider=fallback_provider).get_client(model_config["model"])

    # Real (possibly empty) value, kept separate from the placeholder text used in the prompt -
    # this is what _parse_response falls back to if the model doesn't follow the format, so a
    # parse failure on a no-fold call can never fabricate "(none yet...)" as a real summary.
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
