"""LLM-driven markdown report composer, used by ReportingTools.generate_report.

Given a title/objective and a `context` string, this asks a model to write one well-structured,
substantial markdown report and hands back the raw text. ReportingTools still owns writing it to
disk (see generate_report there); this module only composes the content.

`context` is not just the orchestrator's own short write-up - tools/orchestrator/
orchestrator_tools.py's generate_report auto-appends the full InvestigationState trace (every
tool call and finding made during the investigation) before calling compose_report_markdown, so
the model here typically has the complete raw material for the investigation to draw from, not a
compressed summary of it - hence the length/detail expectations in _SYSTEM_PROMPT below.

Deliberately NOT given any tools or data access of its own: it only has what's in `context`, by
design - the orchestrator/investigation trace is the one holding the real findings, and this call
is instructed to work only from what it's given rather than inventing numbers to make a
"complete-looking" report.
"""
from __future__ import annotations

import logging

from config import get_settings
from llm_provider import LLMProvider
from tools.llm_call import ask_llm_async

logger = logging.getLogger("tools.reporting.report_writer")

DEFAULT_MAX_CONTEXT_CHARS = 24000

_SYSTEM_PROMPT = """You are a report writer for a data analysis platform. You turn already-\
gathered findings into ONE polished, well-structured, THOROUGH markdown report - you do not \
analyze data yourself and must not invent numbers, facts, or findings that are not present in \
the context you're given below. The context you're given is the full trace of everything \
gathered during this investigation - every tool call, agent finding, and number produced - not \
just a short summary, so you have real material to write a substantial report from. Use as much \
of it as is relevant; do not compress a rich context down into a few thin bullet points.

Target length: roughly 1.5-2 pages of a normal document (approximately 900-1300 words of body \
text, not counting headers/table syntax). Write in full, explanatory prose within each section -\
 short, telegraphic bullet fragments throughout the report read as thin and unfinished. A report \
built from a rich context should read like a real analyst wrote it: specific, detailed, and \
grounded in the numbers you were given, not a compressed abstract of them.

Write ONLY the markdown report itself - no commentary, no "Here is your report", no markdown \
code fences wrapping the whole thing.

Follow this structure, adapting section count/depth to how much the context actually supports -\
 never pad with generic filler, but never compress rich context into a token-saving summary \
either:

# <Report title>

**Overview** - two to four sentences (not just one) stating what this report covers, the scope \
of data/files/analyses behind it, and its single most important takeaway.

## Key Findings
For EACH distinct finding in the context, write a short paragraph (2-4 sentences), not a single \
terse bullet fragment - state the finding, the specific numbers/names/dates/comparisons behind \
it, and briefly why it matters or what it implies. Use as many findings as the context actually \
supports - if the investigation touched several files, metrics, or questions, give each its own \
paragraph or sub-heading rather than collapsing them into one list.

## Detailed Analysis
A deeper walkthrough of the supporting data: break down results by file, time period, category, \
or whatever dimensions the context provides. Include a markdown table (proper `| col | col |` \
syntax) wherever the context has structured/tabular figures to lay out - use several tables if \
the context covers multiple datasets or breakdowns. Explain what each table shows in surrounding \
prose rather than dropping it in unexplained. Omit only the parts genuinely unsupported by the \
context - never invent a table or numbers just to fill space.

## Caveats & Open Questions
Any limitations, assumptions, or unresolved items - from the context's own open questions if \
given, or genuine gaps in what was investigated (e.g. date ranges covered, files excluded). Skip \
this section only if the context truly gives nothing for it.

## Data Sources
A bulleted list naming every file/analysis/tool call the findings came from, exactly as named in \
the context - never invent a filename, id, or path that wasn't given to you.

Skip a section only when the context has genuinely nothing to support it - not because the \
report is "long enough" already. A context this rich should almost always fill every section \
above."""

_USER_PROMPT = """Report title: {title}

Objective (why this report is being written):
{objective}

Context - everything already known/found, written by the analyst who did the actual work. This \
is your ONLY source of facts, do not go beyond it:
{context}

Write the report now, following the required structure exactly."""


def get_model_config() -> dict:
    settings = get_settings()
    return {
        "provider": settings.get("REPORT_WRITER_PROVIDER", "") or None,
        "model": settings.get("REPORT_WRITER_MODEL", "") or None,
    }


async def compose_report_markdown(
    title: str, objective: str, context: str, llm_provider: LLMProvider = None, model: str = None,
) -> tuple[str, bool]:
    """Returns (markdown_text, ok). ok=False means the LLM call failed, came back empty, or
    raised - callers (ReportingTools.generate_report) should fall back to a deterministic
    template rather than surface a broken/empty report to the user."""
    if llm_provider is None:
        model_config = get_model_config()
        llm_provider = LLMProvider(model_config["provider"])
        model = model or model_config["model"]

    prompt = (
        _SYSTEM_PROMPT
        + "\n\n"
        + _USER_PROMPT.format(
            title=title, objective=objective, context=(context or "")[:DEFAULT_MAX_CONTEXT_CHARS],
        )
    )

    try:
        client = llm_provider.get_client(model)
        raw = await ask_llm_async(client, prompt)
    except Exception:
        logger.exception("report_writer: composition call failed")
        return "", False

    markdown = (raw or "").strip()
    if not markdown:
        logger.warning("report_writer: model returned an empty report")
        return "", False
    return markdown, True
