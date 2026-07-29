import io

import pandas as pd

from ingestion.storage.base import BaseObjectStore, dedupe_columns


class R2ParquetStore(BaseObjectStore):
    def __init__(self, s3_client, bucket: str, prefix: str = "parquet"):
        self.s3 = s3_client
        self.bucket = bucket
        self.prefix = prefix

    def _key(self, path: str) -> str:
        return f"{self.prefix}/{path}"

    def write(self, data: pd.DataFrame, path: str) -> str:
        data = dedupe_columns(data)
        key = self._key(path)
        buffer = io.BytesIO()
        data.to_parquet(buffer, index=False)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.bucket, key)
        return key

    def read(self, ref: str) -> pd.DataFrame:
        buffer = io.BytesIO()
        try:
            self.s3.download_fileobj(self.bucket, ref, buffer)
        except Exception as exc:
            raise FileNotFoundError(f"No parquet object at r2://{self.bucket}/{ref}") from exc
        buffer.seek(0)
        return pd.read_parquet(buffer)

    def exists(self, ref: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=ref)
            return True
        except Exception:
            return False

    def delete(self, ref: str) -> None:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=ref)
        except Exception:
            pass
