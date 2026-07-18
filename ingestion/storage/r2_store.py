"""BaseObjectStore implementation backed by Cloudflare R2 (S3-compatible),
for when IngestionManager needs to write processed Parquet output somewhere
durable/shared instead of the worker's local disk (see ingestion/README.md's
"How to swap the storage backend" section).

NOT currently used by worker_service's default wiring - kept here as a
documented, available extension only. Reason: tools/tabular/tabular_tools.py
does `getattr(storage, "root_dir", None)` to decide whether the Docker
sandbox (PythonSandbox) is available at all, and duckdb_utils.register_view()
calls DuckDB's read_parquet() directly against `output_ref` - both require a
real local filesystem path, not an R2 object key. Swapping this in as-is
would silently disable run_python (the Tabular Agent's core tool). Using it
for real would require either (a) giving this class a `root_dir` that it
keeps synced with R2 (write-through local cache), or (b) making
tabular_tools.py storage-agnostic (fetch-to-temp before registering/
sandboxing). Neither is done here - worker_service uses LocalParquetStore
instead, matching the engine's own default and documented limitation
("local-only storage" in ingestion/README.md).

Deliberately takes an already-configured boto3 S3 client + bucket name via
the constructor rather than reading credentials itself, for whoever picks
this up later: worker_service already builds a client via
shared.storage.get_s3_client() and would inject it here, so R2 credentials
would live in exactly one place (Server/shared/.env) instead of being
duplicated into analyzerEngine/.env.
"""
import io

import pandas as pd

from ingestion.storage.base import BaseObjectStore


class R2ParquetStore(BaseObjectStore):
    def __init__(self, s3_client, bucket: str, prefix: str = "parquet"):
        self.s3 = s3_client
        self.bucket = bucket
        self.prefix = prefix

    def _key(self, path: str) -> str:
        return f"{self.prefix}/{path}"

    def write(self, data: pd.DataFrame, path: str) -> str:
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
