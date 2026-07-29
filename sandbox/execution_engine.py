import contextlib
import io
import json
import logging
import os
import time
import traceback

import duckdb
import pandas as pd
from path_resolver import get_sandbox_path, new_artifact_id, validate_segment

logger = logging.getLogger("sandbox.execution_engine")


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class ExecutionEngine:
    PREVIEW_CAP = 10
    MAX_STDOUT_CHARS = 500

    def __init__(self):
        self.executions = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        logger.info("ExecutionEngine reset (reset #%d, %d execution(s) served before this reset)",
                    self.resets, self.executions)

    @staticmethod
    def _classify_columns(dtypes: dict) -> dict:
        kinds = {}
        for col, dtype in dtypes.items():
            dtype_lower = dtype.lower()
            if "datetime" in dtype_lower or dtype_lower == "date":
                kinds[col] = "datetime"
            elif dtype_lower == "bool":
                kinds[col] = "categorical"
            elif any(t in dtype_lower for t in ("int", "float", "uint", "double", "decimal")):
                kinds[col] = "numeric"
            else:
                kinds[col] = "categorical"
        return kinds

    def execute(self, code: str, tables: dict, workspace_id: str) -> dict:
        self.executions += 1
        execution_id = self.executions
        t0 = time.perf_counter()
        timings = {}
        logger.info("execute #%d starting: workspace=%s tables=%d", execution_id, workspace_id, len(tables))

        workspace_id = validate_segment(workspace_id, "workspace_id")

        t_tables = time.perf_counter()
        dfs = {}
        con = duckdb.connect(database=":memory:")
        per_table_ms = {}
        for table_name, file_id in tables.items():
            t_one = time.perf_counter()
            path = get_sandbox_path(workspace_id, file_id)
            df = pd.read_parquet(path)
            dfs[table_name] = df
            con.register(table_name, df)
            per_table_ms[table_name] = _ms(t_one)
        timings["table_load_ms"] = per_table_ms
        timings["table_load_total_ms"] = _ms(t_tables)

        saved = []

        def describe(df):
            return {
                "columns": [str(c) for c in df.columns],
                "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
                "shape": list(df.shape),
                "null_counts": {str(c): int(n) for c, n in df.isnull().sum().items()},
            }

        def preview(df, n=5):
            n = max(1, min(int(n), 10))
            return json.loads(df.head(n).to_json(orient="records"))

        def sql(query):
            return con.execute(query).df()

        def save(df, name="result"):
            file_id = new_artifact_id(name)
            full_path = get_sandbox_path(workspace_id, file_id)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            df.to_parquet(full_path, index=False)
            dtypes = {str(c): str(df[c].dtype) for c in df.columns}
            saved.append({
                "file_id": file_id,
                "row_count": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "dtypes": dtypes,
                "column_kinds": self._classify_columns(dtypes),
                "preview": preview(df, self.PREVIEW_CAP),
            })
            return file_id

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
                exec(code, namespace)
        except Exception:
            error = traceback.format_exc()[-2000:]
        timings["exec_ms"] = _ms(t_exec)

        stdout_text = buf.getvalue()
        if len(stdout_text) > self.MAX_STDOUT_CHARS:
            stdout_text = stdout_text[:self.MAX_STDOUT_CHARS] + "\n...[stdout truncated]"

        timings["total_runner_ms"] = _ms(t0)

        result = {"stdout": stdout_text, "saved": saved, "error": error, "timings": timings}
        logger.info(
            "execute #%d finished in %.1fms (error=%s, saved=%d)",
            execution_id, timings["total_runner_ms"], bool(error), len(saved),
        )
        return result
