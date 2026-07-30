"""Follow-up question suggestions, produced by the SAME model call that writes the final answer -
no separate LLM round trip. Shared by every agent whose own final text can end up being what the
user actually sees: OrchestratorAgent always, and TabularAgent/DocumentAgent when direct-routed
(their DIRECT_SYSTEM_MESSAGE variant - see tabular/config.py and document/config.py). NOT added
to their TOOL_SYSTEM_MESSAGE variant (used when the Orchestrator delegates to them) - that text
gets reformatted into the Orchestrator's own final answer, so asking for follow-ups there would
just be thrown away.

FOLLOW_UP_INSTRUCTION gets appended to the relevant system prompts, telling the model to end its
FINAL reply with a small marked section. split_follow_up_questions() then pulls that section back
out of the raw text: everything before the marker is the real answer (what Message.content
becomes), everything after is parsed into exactly 2 follow-up questions
(Message.follow_up_questions - see shared/models/message.py).
"""

FOLLOW_UP_MARKER = "FOLLOW_UP_QUESTIONS:"

FOLLOW_UP_INSTRUCTION = f"""
When you give your FINAL answer (the reply that ends this run - not a status update, not
something said before another tool call), end it with exactly this section, on its own line
after your answer, nothing after it:

{FOLLOW_UP_MARKER}
<first follow-up question>
<second follow-up question>

Exactly 2 short, natural follow-up questions the user might want to ask next - things that build
directly on what you just answered (e.g. asking for a chart of the same data, a breakdown by
another dimension, or a comparison to another period). Write each one as if the USER were about
to type it themselves (e.g. "Make a chart of this", "How does this compare to last quarter") -
never phrased as you asking them a question. One per line, no numbering, no bullets, no quotes.
"""


def split_follow_up_questions(text: str) -> tuple[str, list[str]]:
    """Never raises. A reply with no marker (older behavior, a model that ignored the
    instruction, or this being called on a non-final message by mistake) just returns the text
    unchanged with an empty question list."""
    if not text or FOLLOW_UP_MARKER not in text:
        return text, []

    answer, _, tail = text.partition(FOLLOW_UP_MARKER)
    questions: list[str] = []
    for line in tail.strip().splitlines():
        line = line.strip().lstrip("-*•").strip().strip("\"'").strip()
        if line and line not in questions:
            questions.append(line)
        if len(questions) >= 2:
            break
    return answer.strip(), questions
