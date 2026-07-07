import os

from ingestion.file_types.base import BaseIngestor
from ingestion.file_types.json.utils import (
    flatten_records,
    infer_dtypes,
    load_json,
    stringify_lists,
    to_records,
)
from ingestion.models import IngestionResult


class JSONIngestor(BaseIngestor):
    def validate(self, file_path: str) -> bool:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return False
        try:
            data = load_json(file_path)
            return len(to_records(data)) > 0
        except Exception:
            return False

    def extract_metadata(self, file_path: str) -> dict:
        data = load_json(file_path)
        records = to_records(data)
        df = flatten_records(records[:50])
        return {"columns": list(df.columns), "dtypes": infer_dtypes(df)}

    def ingest(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        try:
            data = load_json(file_path)
            records = to_records(data)
            df = flatten_records(records)
            df = stringify_lists(df)

            if self.storage is None:
                raise RuntimeError("no storage backend provided")

            output_ref = self.storage.write(df, f"{workspace_id}/{file_id}.parquet")

            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="success",
                output_ref=output_ref,
                schema_summary={"columns": list(df.columns), "dtypes": infer_dtypes(df)},
                row_count=len(df),
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
