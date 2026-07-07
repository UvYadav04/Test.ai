from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import duckdb


def connect():
    return duckdb.connect(database=":memory:")


def register_view(con, file_id: str, output_ref: str) -> None:
    con.execute(f"CREATE OR REPLACE VIEW {file_id} AS SELECT * FROM read_parquet('{output_ref}')")


def run_query(con, sql: str, row_cap: int = 500, timeout_seconds: int = 15) -> dict:
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
