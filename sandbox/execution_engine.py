"""Runs INSIDE the persistent sandbox container (see sandbox_server.py, which exposes this
class over an HTTP/UDS endpoint) - never on the host. This is the same execution logic that used
to live directly in runner.py's main() when every run_python() call got its own fresh,
one-shot Docker container (see git history): load the assigned parquet tables as DataFrames,
run the model-generated code against a small set of guarded helpers, capture stdout, and return
a structured result - same JSON shape as before.

The container-per-call cost (Docker create/start, Python interpreter boot, `import
pandas/duckdb`) is what the new architecture removes: one container now serves an entire
investigation's worth of run_python() calls via a long-lived Uvicorn process, and this class is
instantiated exactly once inside that process (see sandbox_server.py). What did NOT change is
the per-call semantics below - every execute() call still (re)loads its tables from disk fresh,
builds a brand new DuckDB connection and Python namespace, and never carries any DataFrame,
variable, or `saved` list over from a previous call. That's deliberate, not an oversight: this
container is reused across MANY run_python calls within one investigation (possibly with
different table sets each time, e.g. a later call joins in a table an earlier call never
touched, or reads back something an earlier call's save() just wrote), so caching a table load
or a namespace across calls would either serve stale data (if the file changed under save()) or
leak one call's local variables into a call that never asked for them. Warm reuse here is an
interpreter/import-cache win only, never a data or execution-state win - see SandboxManager's
own docstring for why that isolation matters even within a single investigation, let alone
across two different ones.

Tables are addressed by file_id, never by path - see path_resolver.py (copied into this image
alongside this file, see Dockerfile). `tables` is {table_name: file_id}; this process derives
the on-disk location itself via get_sandbox_path(workspace_id, file_id), the same function
sandbox_manager.py/sandbox_executor.py use host-side to build the request in the first place.
save() below returns a file_id to model-generated code, never a path - the model has no way to
see (and therefore no way to leak, via print() or otherwise) a real filesystem path."""
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
    """One instance per sandbox container process (see sandbox_server.py) - lives for the whole
    lifetime of that container, across every /execute call the SandboxManager routes to it for
    this investigation. Holds no per-call state between execute() calls (see module docstring);
    `executions`/`resets` are just counters for logging/observability, not execution state."""

    PREVIEW_CAP = 10
    MAX_STDOUT_CHARS = 500

    def __init__(self):
        self.executions = 0
        self.resets = 0

    def reset(self) -> None:
        """Handled by the /reset endpoint. There is no cached table/namespace state to actually
        clear (see module docstring) - this exists as an explicit, logged hook for
        SandboxManager/operators to confirm a sandbox is in a known-clean state before being
        handed to a new caller, and as the extension point if execute() ever grows a cache."""
        self.resets += 1
        logger.info("ExecutionEngine reset (reset #%d, %d execution(s) served before this reset)",
                    self.resets, self.executions)

    @staticmethod
    def _classify_columns(dtypes: dict) -> dict:
        """Coarse numeric/datetime/categorical classification per column, derived purely from
        each column's pandas dtype string - cheap, and computed once here (server-side, with the
        real DataFrame dtypes) rather than left for the orchestrator to guess at from a preview.
        This is what lets a later generate_dashboard/generate_csv call pick a sensible
        label_column/value_column without an extra round trip back through the Tabular Agent
        just to ask "which columns are numeric"."""
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
        """tables: {table_name: file_id}, both already validated by the caller (SandboxManager/
        sandbox_client - see sandbox_executor.py's validate_segment calls) before crossing the
        UDS boundary. Re-validated here too (cheap, and this class has no way of knowing a
        future caller inside this same container won't skip that step) via validate_segment
        (inside get_sandbox_path) - the exact same guard runner.py used to apply, just no longer
        the only thing standing between a bad workspace_id/file_id and a path.

        Returns {"stdout": str, "saved": [...], "error": str | None, "timings": {...}} - the
        exact same shape the host side has always expected back from a sandbox run."""
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
            """Persist the FULL DataFrame to a new Parquet file under this workspace and return its
            file_id (a short string, e.g. "revenue_by_region_a1b2c3d4") - NOT a filesystem path.
            Use this file_id (never a path - there isn't one to have) when reporting what you saved.
            Only a small preview of what was saved is recorded for the caller - never the full data.

            The recorded entry also carries dtypes and a column_kinds classification
            (numeric/datetime/categorical per column, see _classify_columns) - this is what lets
            TabularAgent report real schema/chart-friendly metadata back to the orchestrator
            alongside the file_id, instead of just a bare id it has to re-inspect later."""
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
                exec(code, namespace)  # noqa: S102 - sandboxed: no network, capped resources
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
