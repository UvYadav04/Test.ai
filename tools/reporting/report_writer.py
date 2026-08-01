"""LLM-driven markdown report composer, used by ReportingTools.generate_report.

Given a title/objective and a compact `context` string - findings, numbers, chart mentions,
whatever the orchestrator already knows, written in its own words - this asks a model to write
one well-structured markdown report and hands back the raw text. ReportingTools still owns
writing it to disk (see generate_report there); this module only composes the content.

Deliberately NOT given any tools or data access of its own: it only has what's in `context`, by
design - the orchestrator is the one holding the real findings (from invoke_tabular_agent/
invoke_document_agent/thread history), and this call is instructed to work only from what it's
given rather than inventing numbers to make a "complete-looking" report.
"""
from __future__ import annotations

import logging

from config import get_settings
from llm_provider import LLMProvider
from tools.llm_call import ask_llm_async

logger = logging.getLogger("tools.reporting.report_writer")

DEFAULT_MAX_CONTEXT_CHARS = 12000

_SYSTEM_PROMPT = """You are a report writer for a data analysis platform. You turn already-\
gathered findings into ONE polished, well-structured markdown report - you do not analyze data \
yourself and must not invent numbers, facts, or findings that are not present in the context \
you're given below.

Write ONLY the markdown report itself - no commentary, no "Here is your report", no markdown \
code fences wrapping the whole thing.

Follow this structure exactly:

# <Report title>

A one-paragraph **Overview** stating what this report covers and its single most important \
takeaway.

## Key Findings
A bulleted or numbered list of the concrete, specific findings from the context - real numbers, \
names, dates, comparisons. Each point should stand on its own without needing the others for \
context.

## Details
Only if the context contains enough structured/tabular data to justify it: a markdown table \
(proper `| col | col |` syntax) laying out the supporting figures. Omit this section entirely \
if the context is narrative rather than tabular - never invent a table just to fill space.

## Data Sources
A short bulleted list naming the files/analyses the findings came from, exactly as named in the \
context - never invent a filename, id, or path that wasn't given to you.

Skip any section above that the context genuinely doesn't support rather than padding it with \
generic filler text."""

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
