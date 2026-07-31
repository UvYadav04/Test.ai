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
