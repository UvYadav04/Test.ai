"""Runs inside the sandbox container - never on the host. Loads the assigned parquet files as
DataFrames, executes the model-generated code against a small set of guarded helpers, and writes
a structured result.json for the host process to read back. This script has no network access
(the container is started with networking disabled) and can only read/write under /data (a bind
mount of the app's own parquet storage root) and /job (a per-run scratch directory)."""
import contextlib
import io
import json
import os
import re
import traceback
import uuid

import duckdb
import pandas as pd

MANIFEST_PATH = "/job/manifest.json"
RESULT_PATH = "/job/result.json"
OUTPUT_ROOT = "/data"
PREVIEW_CAP = 20
MAX_STDOUT_CHARS = 500


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    dfs = {}
    con = duckdb.connect(database=":memory:")
    for table_name, path in manifest["tables"].items():
        df = pd.read_parquet(path)
        dfs[table_name] = df
        con.register(table_name, df)

    saved = []

    def describe(df):
        """Schema/shape only - column names, dtypes, row/col count, null counts. No row data."""
        return {
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
            "shape": list(df.shape),
            "null_counts": {str(c): int(n) for c, n in df.isnull().sum().items()},
        }

    def preview(df, n=10):
        """Up to n rows (hard-capped at 50) as a list of dicts - use instead of print(df)."""
        n = max(1, min(int(n), 50))
        return json.loads(df.head(n).to_json(orient="records"))

    def sql(query):
        """Run a DuckDB SQL query over the tables in `dfs`, registered under their table_name."""
        return con.execute(query).df()

    def save(df, name="result"):
        """Persist the FULL DataFrame to a new Parquet file under this workspace and return its
        path. Only a small preview of what was saved is recorded for the caller - never the
        full data."""
        safe_name = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))[:60] or "result"
        result_id = f"{safe_name}_{uuid.uuid4().hex[:8]}"
        rel_path = os.path.join(manifest["workspace_id"], f"{result_id}.parquet")
        full_path = os.path.join(OUTPUT_ROOT, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_parquet(full_path, index=False)
        saved.append({
            "output_ref": full_path,
            "row_count": int(len(df)),
            "columns": [str(c) for c in df.columns],
            "preview": preview(df, PREVIEW_CAP),
        })
        return full_path

    namespace = {
        "dfs": dfs,
        "describe": describe,
        "preview": preview,
        "sql": sql,
        "save": save,
        "pd": pd,
        "duckdb": duckdb,
    }

    buf = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(manifest["code"], namespace)  # noqa: S102 - sandboxed: no network, capped resources
    except Exception:
        error = traceback.format_exc()[-2000:]

    stdout_text = buf.getvalue()
    if len(stdout_text) > MAX_STDOUT_CHARS:
        stdout_text = stdout_text[:MAX_STDOUT_CHARS] + "\n...[stdout truncated]"

    result = {"stdout": stdout_text, "saved": saved, "error": error}
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, default=str)


if __name__ == "__main__":
    main()
