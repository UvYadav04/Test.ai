"""Translates autogen ToolCallRequestEvent/ToolCallExecutionEvent pairs into the
InvestigationEvent shape the frontend renders (see shared/models/investigation.py's
InvestigationEvent and Client/src/components/chat/InvestigationTrail.tsx) - one line for what a
tool is about to do, and one line for what it actually came back with, so a live investigation
reads like running commentary instead of a silent wait.

make_tool_event_translator (the one both DocumentAgent and TabularAgent use) keeps the original
"friendly label per tool name" behavior and adds two optional, per-tool extras:

  - request_detail[tool_name](args: dict) -> str | None
      A short suffix appended to the "about to run this" message, built from the tool's OWN call
      arguments (e.g. the actual search query text, or a code preview for run_python).
  - result_detail[tool_name](args: dict, result) -> str | None
      A suffix for a companion "tool_result" event emitted once the call returns, built from the
      tool's OWN return value (e.g. "Found 6 chunks", or a truncated stdout preview). `result` is
      already parsed (see _parse_result) when the tool's return value was parseable, else the raw
      string.

Neither dict needs an entry for every tool the agent has - anything without a specific builder
still gets a generic truncated preview of its raw result (_generic_result_detail), so every tool
call produces *some* result-side event, not just the ones a caller bothered to special-case.

make_tool_call_translator is kept as-is for any caller that only wants the original request-only
behavior (no result events).
"""
import ast
import json

_DEFAULT_PREVIEW_LIMIT = 160


def truncate(text, limit: int = _DEFAULT_PREVIEW_LIMIT) -> str | None:
    """Collapses a possibly-multiline string down to one readable line (or several - see
    truncate_lines below) capped at `limit` characters, ending in "..." when it was cut short.
    None/empty input -> None, so callers can do `detail = truncate(x) and f"...{detail}"`-style
    checks without a separate emptiness check."""
    if not text:
        return None
    collapsed = " ".join(str(text).split())
    if not collapsed:
        return None
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def truncate_lines(text, max_lines: int = 5, line_limit: int = 200) -> str | None:
    """Like truncate(), but preserves line breaks (each line individually capped) instead of
    collapsing to one line - for things that are naturally multi-line and worth showing as such
    (Python source, stdout output). Frontend renders InvestigationEvent.message with
    whitespace-pre-line, so embedded \\n characters here actually show up as separate lines. Adds
    a bare "..." as its own trailing line when either the line count or an individual line got
    cut."""
    if not text:
        return None
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return None
    truncated_line_count = len(lines) > max_lines
    shown = lines[:max_lines]
    shown = [(ln if len(ln) <= line_limit else ln[:line_limit].rstrip() + "...") for ln in shown]
    if truncated_line_count:
        shown.append("...")
    return "\n".join(shown)


def _parse_result(content):
    """Tool results arrive as a string - JSON, or a Python repr, depending on what the tool
    returned and how autogen serialized it - so try both before giving up and treating it as
    plain text. Never raises."""
    if not isinstance(content, str):
        return content
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(content)
        except Exception:
            continue
    return content


def _generic_result_detail(_args: dict, result) -> str | None:
    """Fallback used whenever a tool has no specific result_detail builder - still gives SOME
    signal (a count for list results, the error for a failed dict result, or a plain truncated
    preview) instead of a silent "done"."""
    if isinstance(result, list):
        return f"{len(result)} result(s)" if result else "No results"
    if isinstance(result, dict) and result.get("error"):
        return truncate(f"Error: {result['error']}")
    if isinstance(result, (dict, list)):
        return truncate(json.dumps(result, default=str))
    return truncate(result)


def make_tool_event_translator(
    friendly_names: dict[str, str],
    request_detail: dict | None = None,
    result_detail: dict | None = None,
):
    request_detail = request_detail or {}
    result_detail = result_detail or {}
    # Keyed by tool NAME, not a call id - same simplification agents/timing.py's ToolCallTimer
    # already makes (autogen's events don't reliably surface a stable per-call id we've verified
    # against this codebase's installed version). Two concurrent calls to the SAME tool inside one
    # batch would clobber each other here; harmless (result_detail just falls back to no extra
    # detail for the second one), and no current agent's max_tool_iterations pattern does that.
    pending_args: dict[str, dict] = {}

    def _label(name: str) -> str:
        return friendly_names.get(name, name)

    def translate(event) -> dict | None:
        event_type = type(event).__name__

        if event_type == "ToolCallRequestEvent":
            parts = []
            names = []
            for call in event.content:
                try:
                    args = json.loads(call.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                pending_args[call.name] = args
                names.append(call.name)
                builder = request_detail.get(call.name)
                detail = builder(args) if builder else None
                parts.append(f"{_label(call.name)}: {detail}" if detail else _label(call.name))
            return {"type": "tool_call", "message": "\n".join(parts), "data": {"tools": names}}

        if event_type == "ToolCallExecutionEvent":
            parts = []
            names = []
            for res in event.content:
                names.append(res.name)
                args = pending_args.pop(res.name, {})
                if getattr(res, "is_error", False):
                    parts.append(f"{_label(res.name)} failed: {truncate(res.content) or 'unknown error'}")
                    continue
                result = _parse_result(res.content)
                builder = result_detail.get(res.name, _generic_result_detail)
                detail = builder(args, result)
                parts.append(f"{_label(res.name)} → {detail}" if detail else f"{_label(res.name)} done")
            return {"type": "tool_result", "message": "\n".join(parts), "data": {"tools": names}}

        return None

    return translate


# Original, request-only translator - still used anywhere that hasn't opted into result events.
def make_tool_call_translator(friendly_names: dict[str, str]):
    def translate(event) -> dict | None:
        if type(event).__name__ != "ToolCallRequestEvent":
            return None
        names = [call.name for call in event.content]
        message = "; ".join(friendly_names.get(name, name) for name in names)
        return {"type": "tool_call", "message": message, "data": {"tools": names}}

    return translate
