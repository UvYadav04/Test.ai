from datetime import datetime, timezone

from tools.orchestrator.models import FileCatalogEntry


class FileCatalog:
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry) -> None:
        self.entries[entry.file_id] = entry

    def remove_entry(self, file_id: str) -> None:
        self.entries.pop(file_id, None)

    def all(self) -> list:
        return list(self.entries.values())

    def browsable(self) -> list:
        """Every entry EXCEPT the ones is_browsable() hides from proactive discovery
        (list_files/search_files/list_tables/the per-turn catalog brief) - see its docstring.
        Nothing is removed from the catalog itself; get_file_details/_to_tabular_file_ref/
        invoke_document_agent can still resolve a hidden entry by file_id, e.g. when the
        orchestrator got that file_id from a Document Agent's table_ref finding rather than
        by browsing for it."""
        return [e for e in self.entries.values() if is_browsable(e)]


def is_browsable(entry) -> bool:
    """Whether a catalog entry should be surfaced through list_files/search_files/list_tables
    and OrchestratorAgent._context_brief's per-turn catalog summary, as opposed to being
    internal-only (still resolvable by file_id elsewhere - see FileCatalog.browsable).

    Two things are hidden:

    - A multi-table source file's own main entry once it has no queryable data of its own:
      an xlsx workbook's output_ref=="" (a workbook has no single "whole file" table - see
      xlsx_ingestor.py) - previously this sailed straight into the catalog brief looking like
      any ordinary file, and the orchestrator picked it directly for invoke_tabular_agent,
      which only failed several turns later deep inside the Docker sandbox with a cryptic
      "not under the sandbox's data root" error (see _to_tabular_file_ref's guard, which this
      makes mostly unnecessary to ever trigger - kept as a defensive fallback, not removed).
      PDF/TXT main entries are the same output_ref shape (a vector-store pointer, not a real
      parquet path) but stay visible: invoke_document_agent operates on THAT exact file_id, so
      it must stay discoverable.

    - A PDF's own per-page table entries (file_type "table", tagged "from_pdf" below). A dense
      PDF can have dozens of these, and _context_brief recomputes the whole catalog summary
      fresh on EVERY investigation turn in a chat (it's baked into every task message, see
      OrchestratorAgent._context_brief) - pre-listing every one would blow up context for that
      turn and, via Chat.summary/recent_turns, bleed into every later turn in the same chat
      too. These are meant to be discovered lazily, one at a time, via a Document Agent's own
      table_ref (see agents/document/config.py) - never pre-enumerated. xlsx's per-sheet table
      entries ("from_xlsx") are the opposite case: there's no "xlsx agent" to discover them
      lazily the way Document Agent does for PDFs, so those are the ONLY queryable surface for
      xlsx data and must stay visible.
    """
    if entry.file_type == "table":
        return "from_xlsx" in (entry.tags or [])
    if entry.file_type in ("pdf", "txt"):
        return True
    return bool(entry.output_ref) and entry.output_ref.endswith(".parquet")


def table_catalog_entry(table: dict, *, source_id: str, source_filename: str,
                         source_file_type: str, uploaded_at) -> FileCatalogEntry:
    """Build a FileCatalogEntry for one table pulled out of a multi-table source file -
    PDF's per-page tables today, xlsx's per-sheet tables next. `table` is one dict from
    IngestionResult.extracted_tables (or the equivalent list persisted on File.extracted_tables
    in Mongo); only "file_id", "output_ref", "row_count", "columns" are required.

    "location" is an optional human-readable string ("page 3", "Sheet1 - table 2") used in the
    display filename. Ingestors that don't set it (PDFIngestor today) fall back to "page {page}"
    so existing behavior is unchanged; new ingestors should just set "location" directly instead
    of adding another one-off fallback here.

    This is the single place that turns an extracted-table dict into a catalog entry - both
    entries_from_ingestion() (fresh off an ingest) and investigation.py's _build_catalog()
    (rebuilt from Mongo) call into this instead of each hardcoding their own filename/tag format.
    """
    location = table.get("location") or (
        f"page {table['page']}" if table.get("page") is not None else f"table {table.get('index', '?')}"
    )
    return FileCatalogEntry(
        file_id=table["file_id"],
        filename=f"{source_filename} ({location})",
        file_type="table",
        uploaded_at=uploaded_at,
        size_bytes=0,
        output_ref=table["output_ref"],
        row_count=table.get("row_count"),
        columns=table.get("columns"),
        tags=[f"from_{source_file_type}", f"source:{source_id}"],
    )


def entries_from_ingestion(result, filename: str, file_type: str, size_bytes: int = 0) -> list:
    """Build FileCatalogEntry objects from an IngestionResult - the main file, plus one entry
    per table a multi-table ingestor (pdf, xlsx) extracted, via table_catalog_entry(). Call this
    after ingest_file() and add each returned entry to a FileCatalog so list_files/list_tables
    can see them."""
    now = datetime.now(timezone.utc)
    entries = [
        FileCatalogEntry(
            file_id=result.file_id,
            filename=filename,
            file_type=file_type,
            uploaded_at=now,
            size_bytes=size_bytes,
            output_ref=result.output_ref,
            row_count=result.row_count,
            page_count=result.schema_summary.get("page_count"),
            columns=result.schema_summary.get("columns"),
        )
    ]

    entries.extend(
        table_catalog_entry(
            table,
            source_id=result.file_id,
            source_filename=filename,
            source_file_type=file_type,
            uploaded_at=now,
        )
        for table in result.extracted_tables
    )

    return entries
