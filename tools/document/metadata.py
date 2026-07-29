from tools.document.token_estimate import estimate_tokens

_MAX_SECTIONS_SHOWN = 20


def build_document_metadata_brief(catalog, vector_store, file_ids: list) -> str:
    if not file_ids:
        return ""

    blocks = []
    for file_id in file_ids:
        entry = catalog.entries.get(file_id) if catalog is not None else None
        chunks = vector_store.get_by_filter({"file_id": file_id})
        table_chunks = [c for c in chunks if c.metadata.get("type") == "table"]
        prose_chunks = [c for c in chunks if c.metadata.get("type") != "table"]

        sections = []
        seen = set()
        for c in prose_chunks:
            title = c.metadata.get("section", "")
            if title and title not in seen:
                seen.add(title)
                sections.append(title)

        estimated_tokens = sum(estimate_tokens(c.text) for c in chunks)

        lines = [f"- file_id: {file_id}"]
        if entry is not None:
            lines.append(f"  filename: {entry.filename}")
            lines.append(f"  file_type: {entry.file_type}")
            if entry.page_count is not None:
                lines.append(f"  page_count: {entry.page_count}")
        lines.append(f"  chunk_count: {len(chunks)}")
        lines.append(f"  estimated_token_count: ~{estimated_tokens}")
        lines.append(f"  table_count: {len(table_chunks)}")
        if sections:
            shown = sections[:_MAX_SECTIONS_SHOWN]
            more = f" (+{len(sections) - _MAX_SECTIONS_SHOWN} more)" if len(sections) > _MAX_SECTIONS_SHOWN else ""
            lines.append(f"  section_headings: {', '.join(shown)}{more}")
        else:
            lines.append("  section_headings: none detected")

        blocks.append("\n".join(lines))

    return "\n".join(blocks)
