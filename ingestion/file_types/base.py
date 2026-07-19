
import os
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ingestion.models import IngestionResult
from ingestion.storage.base import BaseObjectStore
from vectordb.base import BaseVectorStore


def infer_dtypes(df: pd.DataFrame) -> dict:
    """Shared by every ingestor that produces a dataframe (csv, json, xlsx, future txt) -
    csv/utils.py and json/utils.py used to each define their own identical copy of this."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


class BaseIngestor(ABC):

    def __init__(
        self,
        storage: Optional[BaseObjectStore] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self.storage = storage
        self.vector_store = vector_store

    @abstractmethod
    def validate(self, file_path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract_metadata(self, file_path: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def ingest(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        raise NotImplementedError


class SingleTableIngestor(BaseIngestor):
    """Base for ingestors whose output is exactly one dataframe (csv, json, and any future
    flat/text format - tsv, txt-as-table, etc). It implements the validate -> extract_metadata
    -> ingest wiring (read, write to parquet, wrap in an IngestionResult, catch+report errors
    as a "failed" result instead of raising) once, so a new single-table format only has to
    implement `_read_dataframe()` plus whichever hooks it needs below.

    Multi-table formats (pdf, xlsx) don't fit this shape - they extract N tables from one file
    - and should keep implementing BaseIngestor directly; see PDFIngestor's extracted_tables
    handling and IngestionManager/file_catalog.table_catalog_entry for that path.

    Hooks a subclass can override:
      _read_dataframe(file_path, nrows=None) - required. `nrows` is set to a small number
          during validate()/extract_metadata() so those stay cheap on large files.
      _postprocess(df) - format-specific cleanup applied only before the final parquet write
          (e.g. JSON stringifying list/dict columns so parquet can store them).
      _metadata_extra(file_path) - extra keys merged into extract_metadata()'s return dict
          beyond columns/dtypes (e.g. CSV's detected delimiter).
    """

    def _read_dataframe(self, file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        raise NotImplementedError

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _metadata_extra(self, file_path: str) -> dict:
        return {}

    def validate(self, file_path: str) -> bool:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return False
        try:
            df = self._read_dataframe(file_path, nrows=5)
            return len(df.columns) > 0
        except Exception:
            return False

    def extract_metadata(self, file_path: str) -> dict:
        df = self._read_dataframe(file_path, nrows=50)
        return {
            "columns": list(df.columns),
            "dtypes": infer_dtypes(df),
            **self._metadata_extra(file_path),
        }

    def ingest(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        try:
            df = self._postprocess(self._read_dataframe(file_path))

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
