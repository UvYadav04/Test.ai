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
