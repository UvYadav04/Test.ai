from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import duckdb

from sandbox.path_resolver import get_parquet_path, get_table_path

# Re-exported for backward compatibility - get_table_path (sandbox/path_resolver.py) is now the
# single canonical (file_id -> SQL identifier) mapping, shared with ArtifactStore-style callers.
safe_view_name = get_table_path


def connect():
    return duckdb.connect(database=":memory:")


def register_view(con, file_id: str, workspace_id: str, root_dir: str) -> str:
    """Registers a DuckDB view over the parquet artifact identified by (workspace_id, file_id)
    - never a caller-supplied path. The path interpolated into SQL below is always one this
    process built itself from validated ids (get_parquet_path raises InvalidArtifactIdError on
    anything else), not a string handed in from a tool call or a database record."""
    view_name = get_table_path(file_id)
    parquet_path = get_parquet_path(root_dir, workspace_id, file_id)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{parquet_path}')")
    return view_name


def run_query(con, sql: str, row_cap: int = 500, timeout_seconds: int = 15) -> dict:
    # Strip a trailing semicolon (and surrounding whitespace) - the query gets wrapped in an
    # outer SELECT below, and any writer of ordinary, valid SQL will terminate a statement with
    # ";" by habit. Without this, that ordinary style causes a hard parser error on the wrapper.
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()

    def _run():
        total_rows = con.execute(f"SELECT COUNT(*) FROM ({sql}) AS _sub").fetchone()[0]
        result = con.execute(f"SELECT * FROM ({sql}) AS _sub LIMIT {row_cap}")
        columns = [d[0] for d in result.description]
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
        return columns, rows, total_rows

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            columns, rows, total_rows = pool.submit(_run).result(timeout=timeout_seconds)

        return {
            "columns": columns,
            "rows": rows,
            "row_count": total_rows,
            "truncated": total_rows > len(rows),
            "error": None,
        }
    except FutureTimeoutError:
        return {"columns": [], "rows": [], "row_count": 0, "truncated": False, "error": "query timed out"}
    except Exception as exc:
        return {"columns": [], "rows": [], "row_count": 0, "truncated": False, "error": str(exc)}
