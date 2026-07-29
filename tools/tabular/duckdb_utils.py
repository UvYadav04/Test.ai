from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import duckdb

from sandbox.path_resolver import get_parquet_path, get_table_path

safe_view_name = get_table_path


def connect():
    return duckdb.connect(database=":memory:")


def register_view(con, file_id: str, workspace_id: str, root_dir: str) -> str:
    view_name = get_table_path(file_id)
    parquet_path = get_parquet_path(root_dir, workspace_id, file_id)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{parquet_path}')")
    return view_name


def run_query(con, sql: str, row_cap: int = 500, timeout_seconds: int = 15) -> dict:
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
