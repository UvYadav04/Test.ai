"""Generic capability registry for the orchestrator's dynamic tool injection.

The orchestrator exposes a small, fixed set of CORE_TOOLS on every LLM call (list files,
invoke tabular/document agent, ...) and keeps everything else hidden behind named
"capabilities" until the model itself asks for one via `next_capabilities` - see
agents/orchestrator/agent.py's `_wrap_with_next_capabilities` (adds the `next_capabilities`
argument generically to every tool call) and its `run()` outer loop (reads the requested
capability names after each turn and rebuilds the exposed tool list for the next one).

Registering a new capability - a future "presentation", "email", whatever - is exactly one new
entry in CAPABILITY_TOOLS below. Nothing else in the injection mechanism needs to change: the
tool becomes available for the model to request, and its (possibly long) docstring stops
costing prompt tokens on every turn it isn't needed.
"""

# Always exposed, every orchestrator LLM call. Keep this to genuinely core/orientation/
# delegation tools - anything else belongs in CAPABILITY_TOOLS instead, gated behind a request.
CORE_TOOLS = [
    "get_current_date",
    "recall_user_info",
    "list_files",
    "search_files",
    "get_file_details",
    "list_tables",
    "list_file_formats",
    "generate_hypotheses",
    "invoke_tabular_agent",
    "invoke_document_agent",
    "invoke_document_processor",
]

# capability name -> {"tools": [OrchestratorTools method names], "description": one-line
# summary used to build the `next_capabilities` doc every tool call carries - see
# capability_catalog_text() below.
CAPABILITY_TOOLS: dict[str, dict] = {
    "csv": {
        "tools": ["generate_csv"],
        "description": "export an existing data artifact as a CSV file",
    },
    "report": {
        "tools": ["generate_markdown_report"],
        "description": "write a synthesized markdown report file",
    },
    # "dashboard" (generate_dashboard - persistent/auto-refreshing dashboards) is DISABLED for
    # now: invoke_tabular_agent's own create_visualizations tool already covers every chart need,
    # and the real-time/refreshable dashboard feature is redundant on top of that at the moment.
    # Uncomment to re-enable - orchestrator_tools.generate_dashboard and the underlying
    # ReportingTools.generate_realtime_dashboard_bundle/worker_service refresh_dashboard job are
    # both left fully intact, just unreachable while this entry stays commented out.
    # "dashboard": {
    #     "tools": ["generate_dashboard"],
    #     "description": (
    #         "build a persistent, auto-refreshing dashboard from the most recent tabular "
    #         "analysis - NOT for an ordinary chart, which invoke_tabular_agent already produces "
    #         "on its own"
    #     ),
    # },
}


def all_tool_names() -> list[str]:
    """Every tool name the orchestrator could ever expose - core plus every registered
    capability's - used once at OrchestratorAgent construction time to build the full set of
    `next_capabilities`-wrapped callables (see agent.py.__init__)."""
    names = list(CORE_TOOLS)
    for meta in CAPABILITY_TOOLS.values():
        for name in meta["tools"]:
            if name not in names:
                names.append(name)
    return names


def tools_for_capabilities(capability_names) -> list[str]:
    """Tool names to additionally expose for a given list of requested capability names,
    deduped and in registration order. An unknown/hallucinated capability name is silently
    ignored - it should degrade to "no extra tools", not crash the run."""
    names: list[str] = []
    for cap in capability_names or []:
        for name in CAPABILITY_TOOLS.get(cap, {}).get("tools", []):
            if name not in names:
                names.append(name)
    return names


def capability_catalog_text() -> str:
    """Human-readable listing of every registered capability and what it unlocks - this is the
    model's only way to discover valid capability names, since the tools they unlock are hidden
    by default. Embedded into every tool's own `next_capabilities` doc, see agent.py."""
    return "\n".join(f'- "{name}": {meta["description"]}' for name, meta in CAPABILITY_TOOLS.items())
