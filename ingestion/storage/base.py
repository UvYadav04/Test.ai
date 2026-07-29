from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
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
