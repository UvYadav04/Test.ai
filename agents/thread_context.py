def thread_context_brief(thread_context: dict | None) -> str:
    if not thread_context:
        return "This is the first message in this chat - no earlier context."

    lines = []

    summary = thread_context.get("summary")
    if summary:
        lines.append(f"Summary of this chat so far: {summary}")

    recent_turns = thread_context.get("recent_turns") or []
    if recent_turns:
        lines.append(
            "Most recent turns in this chat (oldest first) - use these to resolve references "
            "like \"that file\", \"this data\", \"the same but by region\", or a correction to "
            "what you said before:"
        )
        for turn in recent_turns:
            lines.append(f"- User: {turn.get('query', '')}")
            lines.append(f"  You answered: {turn.get('response', '')}")

    files_used = thread_context.get("files_used") or []
    if files_used:
        lines.append(f"file_ids already queried earlier in this chat: {files_used}")

    files_created = thread_context.get("files_created") or []
    if files_created:
        lines.append(f"Artifacts already produced earlier in this chat: {files_created}")

    return "\n".join(lines) if lines else "This is the first message in this chat - no earlier context."
