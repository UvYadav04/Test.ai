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

CAPABILITY_TOOLS: dict[str, dict] = {
    "csv": {
        "tools": ["generate_csv"],
        "description": "export an existing data artifact as a CSV file",
    },
    "report": {
        "tools": ["generate_report"],
        "description": "write an LLM-composed markdown report file from findings you already have",
    },
}


def all_tool_names() -> list[str]:
    names = list(CORE_TOOLS)
    for meta in CAPABILITY_TOOLS.values():
        for name in meta["tools"]:
            if name not in names:
                names.append(name)
    return names


def tools_for_capabilities(capability_names) -> list[str]:
    names: list[str] = []
    for cap in capability_names or []:
        for name in CAPABILITY_TOOLS.get(cap, {}).get("tools", []):
            if name not in names:
                names.append(name)
    return names


def capability_catalog_text() -> str:
    return "\n".join(f'- "{name}": {meta["description"]}' for name, meta in CAPABILITY_TOOLS.items())
