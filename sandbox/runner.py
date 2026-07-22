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
import time
import traceback
import uuid

import duckdb
import pandas as pd

MANIFEST_PATH = "/job/manifest.json"
RESULT_PATH = "/job/result.json"
OUTPUT_ROOT = "/data"
PREVIEW_CAP = 5
MAX_STDOUT_CHARS = 500


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def main():
    # t0 is captured AFTER the `import duckdb`/`import pandas` at the top of this file have
    # already run - those two imports are commonly 0.5-1.5s cold (numpy/duckdb's native libs
    # loading), on top of the Python interpreter's own boot time. Neither is visible inside
    # this timings dict for that reason: subtract total_runner_ms (below) from however long the
    # HOST saw the container run for (sandbox_executor.py logs this separately) to get that
    # import+interpreter-startup cost by elimination.
    t0 = time.perf_counter()
    timings = {}

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    timings["manifest_load_ms"] = _ms(t0)

    t_tables = time.perf_counter()
    dfs = {}
    con = duckdb.connect(database=":memory:")
    per_table_ms = {}
    for table_name, path in manifest["tables"].items():
        t_one = time.perf_counter()
        df = pd.read_parquet(path)
        dfs[table_name] = df
        con.register(table_name, df)
        per_table_ms[table_name] = _ms(t_one)
    timings["table_load_ms"] = per_table_ms
    timings["table_load_total_ms"] = _ms(t_tables)

    saved = []

    def describe(df):
        """Schema/shape only - column names, dtypes, row/col count, null counts. No row data."""
        return {
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
            "shape": list(df.shape),
            "null_counts": {str(c): int(n) for c, n in df.isnull().sum().items()},
        }

    def preview(df, n=5):
        """Up to n rows (hard-capped at 50) as a list of dicts - use instead of print(df)."""
        n = max(1, min(int(n), 10))
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

    t_exec = time.perf_counter()
    buf = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(manifest["code"], namespace)  # noqa: S102 - sandboxed: no network, capped resources
    except Exception:
        error = traceback.format_exc()[-2000:]
    # Includes every save() call the model-generated code made (to_parquet writes) - those
    # aren't timed separately, they're wherever the code called them inside exec() above.
    timings["exec_ms"] = _ms(t_exec)

    stdout_text = buf.getvalue()
    if len(stdout_text) > MAX_STDOUT_CHARS:
        stdout_text = stdout_text[:MAX_STDOUT_CHARS] + "\n...[stdout truncated]"

    # Captured just before the final write, not after - so it reflects "time spent doing work",
    # not the write itself. sandbox_executor.py logs this alongside its own host-side timings.
    timings["total_runner_ms"] = _ms(t0)

    result = {"stdout": stdout_text, "saved": saved, "error": error, "timings": timings}
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, default=str)


if __name__ == "__main__":
    main()
