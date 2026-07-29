def make_tool_call_translator(friendly_names: dict[str, str]):
    
    def translate(event) -> dict | None:
        if type(event).__name__ != "ToolCallRequestEvent":
            return None
        names = [call.name for call in event.content]
        message = "; ".join(friendly_names.get(name, name) for name in names)
        return {"type": "tool_call", "message": message, "data": {"tools": names}}

    return translate
