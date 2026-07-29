from typing import Optional

import pandas as pd

from ingestion.file_types.base import SingleTableIngestor
from ingestion.file_types.csv.utils import detect_delimiter


class CSVIngestor(SingleTableIngestor):
    def _read_dataframe(self, file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        return pd.read_csv(file_path, sep=detect_delimiter(file_path), nrows=nrows)

    def _metadata_extra(self, file_path: str) -> dict:
        return {"delimiter": detect_delimiter(file_path)}
