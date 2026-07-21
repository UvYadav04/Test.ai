from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet (via pyarrow) requires unique column names - pandas itself doesn't enforce
    that, so a DataFrame from a genuinely messy source can reach a store's write() with
    duplicates and blow up to_parquet() with "Duplicate column names found: [...]". The
    PDF pipeline is the main source of this: docling's table export for a financial table
    whose header row(s) collapsed to repeated values (e.g. a multi-tier header where two
    sub-columns both display "33" or "2%" - see PDFIngestor._extract_tables) produces
    exactly this. Renames every repeat with a numeric suffix (col, col__1, col__2, ...)
    rather than rejecting the table outright - the data underneath is still useful even
    when a couple of header labels weren't unique in the source."""
    if df.columns.duplicated().any():
        counts: dict = {}
        new_columns = []
        for col in df.columns:
            if col not in counts:
                counts[col] = 0
                new_columns.append(col)
            else:
                counts[col] += 1
                new_columns.append(f"{col}__{counts[col]}")
        df = df.copy()
        df.columns = new_columns
    return df


class BaseObjectStore(ABC):
    @abstractmethod
    def write(self, data: Any, path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def read(self, ref: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def exists(self, ref: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete(self, ref: str) -> None:
        raise NotImplementedError
