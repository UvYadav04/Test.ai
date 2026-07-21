import os

from ingestion.file_types.base import BaseIngestor, infer_dtypes
from ingestion.file_types.xlsx.utils import detect_tables, load_sheets
from ingestion.models import IngestionResult


class XLSXIngestor(BaseIngestor):
    """Multi-table format - a workbook can hold several sheets, and a sheet can hold several
    stacked/side-by-side tables (see xlsx/utils.py) - so it doesn't fit SingleTableIngestor's
    one-dataframe shape. Follows PDFIngestor's extracted_tables pattern instead: there's no
    single "primary" table, every detected table gets its own parquet write and its own entry
    in IngestionResult.extracted_tables (the same mechanism table_catalog_entry() already
    generalized for pdf). No vector_store involved - like csv/json this is pure tabular data,
    no free text to chunk/embed. Charts, images, and embedded objects are intentionally not
    read - only cell values."""

    def __init__(self, storage=None, vector_store=None):
        super().__init__(storage=storage, vector_store=vector_store)
        self.errors = []

    def validate(self, file_path: str) -> bool:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return False
        try:
            sheets = load_sheets(file_path)
            if not sheets:
                self.errors = ["workbook has no sheets"]
                return False
            if not any(detect_tables(grid) for grid in sheets.values()):
                self.errors = ["no tables detected in any sheet"]
                return False
            return True
        except Exception as exc:
            self.errors = [str(exc)]
            return False

    def extract_metadata(self, file_path: str) -> dict:
        sheets = load_sheets(file_path)
        tables_meta = []
        for sheet_name, grid in sheets.items():
            for index, table in enumerate(detect_tables(grid)):
                dataframe = table["dataframe"]
                tables_meta.append({
                    "sheet": sheet_name,
                    "index": index,
                    "columns": list(dataframe.columns),
                    "dtypes": infer_dtypes(dataframe),
                    "row_count": len(dataframe),
                })
        return {
            "sheet_count": len(sheets),
            "table_count": len(tables_meta),
            "sheets": list(sheets.keys()),
            "tables": tables_meta,
        }

    def ingest(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        try:
            if self.storage is None:
                raise RuntimeError("no storage backend provided")

            sheets = load_sheets(file_path)
            extracted_tables = []

            for sheet_name, grid in sheets.items():
                tables = detect_tables(grid)
                multi = len(tables) > 1
                for sheet_index, table in enumerate(tables):
                    dataframe = table["dataframe"]
                    table_file_id = f"{file_id}_table_{len(extracted_tables)}"

                    output_ref = self.storage.write(dataframe, f"{workspace_id}/{table_file_id}.parquet")
                    columns = [str(c) for c in dataframe.columns]
                    location = f"{sheet_name} - table {sheet_index + 1}" if multi else sheet_name

                    extracted_tables.append({
                        "file_id": table_file_id,
                        "output_ref": output_ref,
                        "row_count": len(dataframe),
                        "columns": columns,
                        "location": location,
                        "sheet": sheet_name,
                        "index": sheet_index,
                    })

            errors = [] if extracted_tables else ["no tables detected in any sheet"]
            status = "success" if extracted_tables else "failed"

            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status=status,
                output_ref="",
                schema_summary={
                    "sheet_count": len(sheets),
                    "table_count": len(extracted_tables),
                    "sheets": list(sheets.keys()),
                },
                extracted_tables=extracted_tables,
                errors=errors,
            )
        except Exception as exc:
            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="failed",
                output_ref="",
                schema_summary={},
                errors=[str(exc)],
            )
