import os

import pandas as pd

from ingestion.storage.base import BaseObjectStore


class LocalParquetStore(BaseObjectStore):
    def __init__(self, root_dir: str = "data/parquet"):
        print("root dir: ",root_dir)
        self.root_dir = root_dir

    def write(self, data: pd.DataFrame, path: str) -> str:
        print("path: ",path)
        full_path = os.path.join(self.root_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        data.to_parquet(full_path, index=False)
        return full_path

    def read(self, ref: str) -> pd.DataFrame:
        return pd.read_parquet(ref)

    def exists(self, ref: str) -> bool:
        return os.path.exists(ref)

    def delete(self, ref: str) -> None:
        if os.path.exists(ref):
            os.remove(ref)
