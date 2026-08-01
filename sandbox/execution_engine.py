import contextlib
import gc
import glob
import io
import json
import logging
import multiprocessing
import os
import shutil
import signal
import sys
import threading
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
    THREAD_JOIN_TIMEOUT_S = 1.0

    def __init__(self):
        self.executions = 0
        self.resets = 0
        self._healthy = True
        # Threads alive when this (long-lived, one-per-pooled-container) engine was created -
        # i.e. uvicorn/asyncio's own worker threads. Anything alive at reset() time that isn't
        # in this set was spawned by exec()'d user code and didn't clean up after itself.
        self._baseline_thread_idents = {t.ident for t in threading.enumerate()}

    @property
    def healthy(self) -> bool:
        """Whether the last execute()/reset() cycle left the sandbox in a state a pool
        manager should trust for the next caller. False means: discard this container."""
        return self._healthy

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
        try:
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
        finally:
            # The namespace/dfs/duckdb-connection above are already fresh per call and go out
            # of scope when execute() returns, so they can't leak into the *next* execution on
            # their own - but the DuckDB connection is a real OS-level resource (file
            # descriptors, worker threads) that needs an explicit close rather than relying on
            # __del__/GC timing, especially now that a container serves many calls over its
            # lifetime instead of just one.
            try:
                con.close()
            except Exception:
                logger.exception("execute #%d: failed to close duckdb connection", execution_id)

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

    def reset(self) -> bool:
        """Defensive cleanup run between pooled executions, before this sandbox is handed to
        the next caller. execute() already builds a fresh namespace + DuckDB connection every
        call, so this mainly catches what user code can leak *outside* that per-call scope:
        background threads, child processes, matplotlib figures, and stray files under /tmp.

        Returns False if the sandbox couldn't be fully cleaned - the pool manager treats that
        as a signal to discard this container instead of reusing it.
        """
        self.resets += 1
        ok = True
        ok &= self._join_leaked_threads()
        ok &= self._clear_matplotlib()
        ok &= self._kill_child_processes()
        ok &= self._clean_tmp_files()
        gc.collect()
        self._healthy = ok
        logger.info(
            "ExecutionEngine reset #%d (%d execution(s) served so far, healthy=%s)",
            self.resets, self.executions, ok,
        )
        return ok

    def _join_leaked_threads(self) -> bool:
        leaked = [
            t for t in threading.enumerate()
            if t.ident not in self._baseline_thread_idents and t is not threading.current_thread()
        ]
        if not leaked:
            return True
        ok = True
        for t in leaked:
            t.join(timeout=self.THREAD_JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning(
                    "reset: background thread %r spawned by executed code is still running "
                    "%.1fs after exec() returned - Python can't force-kill threads, so this "
                    "sandbox is being marked unhealthy for discard instead of reuse",
                    t.name, self.THREAD_JOIN_TIMEOUT_S,
                )
                ok = False
            else:
                logger.info("reset: joined a leftover thread (%r) left running after exec()", t.name)
        return ok

    @staticmethod
    def _clear_matplotlib() -> bool:
        plt = sys.modules.get("matplotlib.pyplot")
        if plt is None:
            return True
        try:
            plt.close("all")
            return True
        except Exception:
            logger.exception("reset: failed to clear matplotlib figures")
            return False

    @staticmethod
    def _kill_child_processes() -> bool:
        ok = True
        try:
            for child in list(multiprocessing.active_children()):
                child.terminate()
                child.join(timeout=1.0)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=1.0)
        except Exception:
            logger.exception("reset: failed to reap multiprocessing children")
            ok = False

        # Executed code can also shell out via subprocess/os.system without going through
        # multiprocessing at all. On Linux (this container always is) /proc lets us find and
        # kill anything still parented to this process without needing psutil.
        try:
            my_pid = os.getpid()
            for stat_path in glob.glob("/proc/[0-9]*/stat"):
                try:
                    with open(stat_path) as f:
                        content = f.read()
                    # Format: "pid (comm) state ppid ...". comm can itself contain spaces or
                    # parens (e.g. a process renamed via setproctitle), so field-splitting the
                    # whole line would misalign everything after it - split only on what comes
                    # after the last ')' instead.
                    rest = content[content.rindex(")") + 1:].split()
                    ppid = int(rest[1])
                    if ppid != my_pid:
                        continue
                    child_pid = int(stat_path.split("/")[2])
                    os.kill(child_pid, signal.SIGKILL)
                    logger.warning("reset: killed leaked child process pid=%d", child_pid)
                except (OSError, ValueError, IndexError):
                    continue
        except Exception:
            logger.exception("reset: failed to scan /proc for leaked child processes")
            ok = False
        return ok

    @staticmethod
    def _clean_tmp_files() -> bool:
        ok = True
        for path in glob.glob("/tmp/*"):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
            except OSError:
                logger.warning("reset: could not remove leftover temp path %s", path)
                ok = False
        return ok
