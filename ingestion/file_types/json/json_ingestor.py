import os
from typing import Optional

import pandas as pd

from ingestion.file_types.base import SingleTableIngestor
from ingestion.file_types.json.utils import (
    flatten_records,
    load_json,
    stringify_lists,
    to_records,
)


class JSONIngestor(SingleTableIngestor):
    """extract_metadata/ingest come from SingleTableIngestor - this class supplies the
    JSON-specific bits: record extraction/flattening, stringifying list/dict columns so
    parquet can store them, and its own validate() (JSON validity is "has at least one
    record", not "has at least one column" - a record can flatten to zero columns and still
    be a real row, unlike CSV/base's default column-count check)."""

    def validate(self, file_path: str) -> bool:
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return False
        try:
            return len(to_records(load_json(file_path))) > 0
        except Exception:
            return False

    def _read_dataframe(self, file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        records = to_records(load_json(file_path))
        if nrows is not None:
            records = records[:nrows]
        return flatten_records(records)

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return stringify_lists(df)

    def _metadata_extra(self, file_path: str) -> dict:
        return {"top_level_type": "list" if isinstance(load_json(file_path), list) else "dict"}
