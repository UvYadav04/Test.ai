FOLLOW_UP_MARKER = "FOLLOW_UP_QUESTIONS:"

FOLLOW_UP_INSTRUCTION = f"""
When you give your FINAL answer (the reply that ends this run - not a status update, not
something said before another tool call), end it with exactly this section, on its own line
after your answer, nothing after it:

{FOLLOW_UP_MARKER}
<first follow-up suggestion>
<second follow-up suggestion>

Exactly 2 short follow-up suggestions for what the user might want done next. These are sent
back to you VERBATIM as the user's next message if they're clicked - so phrase each one as a
direct request/instruction in the user's voice (imperative, e.g. "Create a line chart showing
the monthly sales trend for each product"), NEVER as a yes/no question asking for permission or
your opinion (e.g. NOT "Should I create a line chart...?" or "Would a line chart help?"). A
suggestion phrased as a question gets misread as "give me your opinion on whether to do this"
instead of "go do this" - always write the action you want performed, not a question about it.
"""

import re

FOLLOW_UP_PATTERN = re.compile(
    r"follow[\s\-_-–—]*up[\s\-_-–—]*questions?\s*:",
    re.IGNORECASE,
)


def split_follow_up_questions(text: str) -> tuple[str, list[str]]:
    if not text:
        return text, []

    match = FOLLOW_UP_PATTERN.search(text)
    if not match:
        return text, []

    answer = text[:match.start()].strip()
    tail = text[match.end():]

    questions: list[str] = []
    for line in tail.splitlines():
        line = line.strip()
        line = re.sub(r"^[-*•\d.)\s]+", "", line)  # Remove bullets/numbers
        line = line.strip("\"'")

        if line and line not in questions:
            questions.append(line)

        if len(questions) >= 2:
            break

    return answer, questions