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


def entries_from_ingestion(result, filename: str, file_type: str, size_bytes: int = 0) -> list:
    """Build FileCatalogEntry objects from an IngestionResult - the main file, plus one entry
    per table a PDF's hybrid pipeline extracted (file_type="table", tagged with the source
    file_id). Call this after ingest_file() and add each returned entry to a FileCatalog so
    list_files/list_tables can see them."""
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

    for table in result.extracted_tables:
        entries.append(FileCatalogEntry(
            file_id=table["file_id"],
            filename=f"{filename} (table, page {table['page']})",
            file_type="table",
            uploaded_at=now,
            size_bytes=0,
            output_ref=table["output_ref"],
            row_count=table["row_count"],
            columns=table["columns"],
            tags=["from_pdf", f"source:{result.file_id}"],
        ))

    return entries
