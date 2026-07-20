"""Shared formatting for Chat thread continuity (see worker_service/tasks/investigation.py's
_thread_context - summary/recent_turns/files_used/files_created read off the Chat doc) into a
prompt-ready brief.

Used by every agent that can end up being the one actually producing the final answer for a
turn: OrchestratorAgent always, and TabularAgent/DocumentAgent too when run_investigation
direct-routes to them (skipping the Orchestrator entirely - see _run_tabular_direct/
_run_document_direct). Without this, a direct-routed agent sees ONLY the current message with no
memory of earlier turns in the same chat, so a follow-up like "show me a chart of this data" -
where "this data" refers to something established a turn or two earlier - has nothing to resolve
the reference against. Centralized here (rather than duplicated per agent) so all three read
"earlier context" the exact same way.
"""


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
