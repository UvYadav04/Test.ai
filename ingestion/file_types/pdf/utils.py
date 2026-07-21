"""Docling-document helpers shared by PDFIngestor. Parsing itself no longer happens here -
that moved to llamaparse_client.py (LlamaParse's cloud API replaced docling's local CPU
conversion, see that module's docstring) - this file now only holds the small utilities that
operate on an already-built DoclingDocument: table extraction/captioning and the
is_scanned() heuristic used per LlamaParse page."""


def is_scanned(document) -> bool:
    text = document.export_to_text()
    return len(text.strip()) < 20


def extract_tables(document, chunks: list = None, page_override: int = None) -> list:
    """`page_override`: when the caller already knows the true page number (llamaparse_client
    builds one DoclingDocument per LlamaParse page, so docling's own provenance - meaningful
    only for native PDF conversion - would just say "page 1" every time), pass it here
    instead of trusting _table_page()'s docling-provenance lookup."""
    tables = []
    for index, table in enumerate(document.tables):
        try:
            dataframe = table.export_to_dataframe(doc=document)
        except Exception:
            continue
        if dataframe.empty:
            continue
        page = page_override if page_override is not None else _table_page(table)
        tables.append({
            "index": index,
            "dataframe": dataframe,
            "page": page,
            "caption": _infer_caption(table, document, chunks, page),
        })
    return tables


def _table_page(table) -> int:
    try:
        return table.prov[0].page_no
    except Exception:
        return 0


def _infer_caption(table, document, chunks: list, page: int) -> str:
    """4-tier fallback: explicit docling caption -> nearby text snippet -> section title -> none."""
    explicit = _explicit_caption(table, document)
    if explicit:
        return explicit

    nearby = _nearest_chunk(chunks, page)
    if nearby is None:
        return f"Table on page {page}"

    snippet = nearby.text.strip().splitlines()[0][:120] if nearby.text.strip() else ""
    if snippet:
        return snippet

    if nearby.section:
        return nearby.section

    return f"Table on page {page}"


def _explicit_caption(table, document) -> str:
    try:
        return table.caption_text(document) or ""
    except Exception:
        return ""


def _nearest_chunk(chunks: list, page: int):
    if not chunks:
        return None
    before = [c for c in chunks if c.page <= page]
    pool = before or chunks
    return max(pool, key=lambda c: c.page)
