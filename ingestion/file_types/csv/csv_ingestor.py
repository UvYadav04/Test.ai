import os

import pandas as pd

from ingestion.file_types.base import BaseIngestor
from ingestion.file_types.csv.utils import detect_delimiter, infer_dtypes
from ingestion.models import IngestionResult


class CSVIngestor(BaseIngestor):
    def validate(self, file_path: str) -> bool:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return False
        try:
            df = pd.read_csv(file_path, nrows=5, sep=detect_delimiter(file_path))
            return len(df.columns) > 0
        except Exception:
            return False

    def extract_metadata(self, file_path: str) -> dict:
        delimiter = detect_delimiter(file_path)
        df = pd.read_csv(file_path, sep=delimiter, nrows=50)
        return {
            "columns": list(df.columns),
            "dtypes": infer_dtypes(df),
            "delimiter": delimiter,
        }

    def ingest(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        try:
            delimiter = detect_delimiter(file_path)
            df = pd.read_csv(file_path, sep=delimiter)

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
