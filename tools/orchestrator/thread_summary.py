"""Maintains Chat.summary - a compact rolling recap of a thread, regenerated
once per completed investigation (see
worker_service.tasks.investigation._update_chat_continuity) so the NEXT
investigation's task prompt can hand the orchestrator a small, plain-
language "here's what's happened in this chat so far" instead of nothing at
all (see OrchestratorAgent._thread_context_brief).

Deliberately its own tiny LLM call rather than something folded into the
orchestrator's own run: it only ever sees {previous_summary, latest user
message, latest final answer} - no tool schemas, no file catalog - so it
stays small and fast regardless of how large the orchestrator's own prompt
grows, and a failure here (see call site) never has to fail the investigation
itself.
"""

from agents.orchestrator.config import get_model_config
from llm_provider import LLMProvider, get_settings
from tools.llm_call import ask_llm_async

SUMMARY_PROMPT = """You maintain a running summary of an ongoing data-analysis conversation, so a
fresh orchestrator agent can pick up a follow-up question knowing what's already been asked and
found - especially when the user refers back to something ("that file", "same but by region") or
corrects an earlier turn ("no, I meant Q2 not Q3").

Previous summary:
{previous_summary}

Latest turn:
User: {query}
Assistant: {response}

Write the updated summary, folding the latest turn into the previous one. Keep it compact plain
language - a few sentences, not a transcript - covering what's been asked and found so far. If
the latest turn corrects or contradicts something in the previous summary, keep only the
corrected version, don't preserve both. Do not include file paths or ids verbatim (those are
tracked separately) and do not add any preamble like "Summary:" - reply with just the summary
text itself."""


async def update_summary(previous_summary: str, query: str, response: str) -> str:
    model_config = get_model_config()
    # See orchestrator/agent.py's comment on FALLBACK_LLM_PROVIDER - same reasoning here.
    fallback_provider = get_settings().get("FALLBACK_LLM_PROVIDER", "groq")
    client = LLMProvider(model_config["provider"], fallback_provider=fallback_provider).get_client(model_config["model"])
    prompt = SUMMARY_PROMPT.format(
        previous_summary=previous_summary or "(none yet - this is the first turn in this chat)",
        query=query,
        response=response,
    )
    return (await ask_llm_async(client, prompt)).strip()
