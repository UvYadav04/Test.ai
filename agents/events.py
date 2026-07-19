"""Shared translation from AutoGen's raw ToolCallRequestEvent into the
compact {type, message, data} shape streamed to the frontend's live
activity panel (see shared.models.investigation.InvestigationEvent).

Two rules apply everywhere this is used (orchestrator AND the delegated
Tabular/Document agents - see their agent.py's `on_event` plumbing):

1. Only "started" events are ever produced - there's deliberately no
   matching "done"/tool_result event. A step finishing doesn't tell the
   user anything new (the next event, or the final answer, already implies
   it), so a second row per step just doubled the trail's length without
   adding information.
2. `message` never names which internal agent is running - callers pass a
   friendly-name map scoped to that agent's own tools, so the user sees
   WHAT is happening ("Executing a Python script"), never WHICH agent
   ("the Tabular Agent") is doing it.
"""


def make_tool_call_translator(friendly_names: dict[str, str]):
    """Returns a `translate(event) -> dict | None` bound to one agent's own
    friendly-name map. Assign the result to that agent's `_translate_event`
    (as a plain callable, e.g. via `staticmethod(...)`)."""

    def translate(event) -> dict | None:
        if type(event).__name__ != "ToolCallRequestEvent":
            return None
        names = [call.name for call in event.content]
        message = "; ".join(friendly_names.get(name, name) for name in names)
        return {"type": "tool_call", "message": message, "data": {"tools": names}}

    return translate
