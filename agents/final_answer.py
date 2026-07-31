FOLLOW_UP_MARKER = "FOLLOW_UP_QUESTIONS:"

FOLLOW_UP_INSTRUCTION = f"""
When you give your FINAL answer (the reply that ends this run - not a status update, not
something said before another tool call), end it with exactly this section, on its own line
after your answer, nothing after it:

{FOLLOW_UP_MARKER}
<first follow-up question>
<second follow-up question>

Exactly 2 short, natural follow-up questions the user might want to ask next.
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