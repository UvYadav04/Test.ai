"""Translates autogen ToolCallRequestEvent/ToolCallExecutionEvent pairs into the
InvestigationEvent shape the frontend renders (see shared/models/investigation.py's
InvestigationEvent and Client/src/components/chat/InvestigationTrail.tsx).

Two events per tool call - "tool_call" when it starts, "tool_result" (success) or "tool_error"
(failure) when it finishes - so the frontend's per-step UI can pair them into one row: a spinner
while the matching result hasn't arrived yet, swapped for a check or cross once it has. See
InvestigationTrail.tsx's buildRows() - it pairs by stream order (each tool_call is immediately
followed by its own tool_result/tool_error) and only ever shows the loader beside whichever
tool_call is still unpaired, i.e. the latest one.

Messages are ONLY the friendly tool name - no query text, no code preview, no result counts or
previews, no `data` payload. Nothing downstream reads more than "which tool, did it succeed" -
the paired UI row doesn't even render the tool_result event's own message text, it reuses the
tool_call's (see buildRows()), so keeping these minimal costs nothing.
"""


def make_tool_event_translator(friendly_names: dict[str, str]):
    def _label(name: str) -> str:
        return friendly_names.get(name, name)

    def translate(event) -> dict | None:
        event_type = type(event).__name__

        if event_type == "ToolCallRequestEvent":
            message = "; ".join(_label(call.name) for call in event.content)
            return {"type": "tool_call", "message": message}

        if event_type == "ToolCallExecutionEvent":
            any_error = any(getattr(res, "is_error", False) for res in event.content)
            message = "; ".join(_label(res.name) for res in event.content)
            return {"type": "tool_error" if any_error else "tool_result", "message": message}

        return None

    return translate
