import re
import uuid
from typing import Optional

from tools.tabular.duckdb_utils import connect, register_view, run_query
from tools.tabular.models import (
    ColumnProfile,
    FileMetadata,
    JoinCandidate,
    MetricSpec,
    QueryResult,
    SchemaInfo,
    ValidationReport,
)


class TabularTools:
    def __init__(self, assigned_files: list, storage=None, workspace_id: str = "default"):
        self.assigned_files = {f.file_id: f for f in assigned_files}
        self.con = connect()
        self.storage = storage
        self.workspace_id = workspace_id
        self.table_names = {}
        for file_ref in assigned_files:
            self.table_names[file_ref.file_id] = register_view(self.con, file_ref.file_id, file_ref.output_ref)

    def _check_assigned(self, file_id: str) -> None:
        if file_id not in self.assigned_files:
            raise ValueError(f"file_id '{file_id}' is not assigned to this agent")

    def _table(self, file_id: str) -> str:
        return self.table_names[file_id]

    @staticmethod
    def _quote_ident(name: str) -> str:
        """Quote a column/alias name for safe use inside SQL we build ourselves - real-world
        CSV headers routinely contain spaces ("Job Title", "User Id") which are not valid
        unquoted SQL identifiers."""
        return '"' + str(name).replace('"', '""') + '"'

    def list_allowed_files(self) -> list:
        """Return metadata (row count, columns, queryable table_name) for every file assigned to
        this agent. file_id may contain characters invalid in SQL (dots, hyphens) - always use
        table_name, not file_id, inside raw SQL you write for query_data."""
        files = []
        for file_id, file_ref in self.assigned_files.items():
            table = self._table(file_id)
            row_count = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            columns = [d[0] for d in self.con.execute(f"SELECT * FROM {table} LIMIT 0").description]
            files.append({
                "file_id": file_id,
                "table_name": table,
                "filename": file_ref.filename,
                "output_ref": file_ref.output_ref,
                "row_count": row_count,
                "columns": columns,
            })
        return files

    def inspect_schema(self, file_id: str) -> SchemaInfo:
        """Return column names, dtypes, nullability, and likely key columns for one assigned file."""
        self._check_assigned(file_id)
        table = self._table(file_id)

        info = self.con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()
        columns = [row[0] for row in info]
        dtypes = {row[0]: row[1] for row in info}
        nullable = {row[0]: row[2] == "YES" for row in info}

        total = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        likely_keys = []
        for col in columns:
            if col.lower().endswith("_id") or col.lower().endswith(" id"):
                likely_keys.append(col)
                continue
            distinct = self.con.execute(
                f"SELECT COUNT(DISTINCT {self._quote_ident(col)}) FROM {table}"
            ).fetchone()[0]
            if total > 0 and distinct >= total * 0.95:
                likely_keys.append(col)

        return {
            "columns": columns,
            "dtypes": dtypes,
            "nullable": nullable,
            "sample_size": total,
            "likely_key_columns": likely_keys,
        }

    def sample_rows(self, file_id: str, n: int = 10) -> list:
        """Return up to n example rows from a file, to check real values before writing a query."""
        self._check_assigned(file_id)
        n = min(n, 50)
        table = self._table(file_id)
        result = self.con.execute(f"SELECT * FROM {table} LIMIT {n}")
        columns = [d[0] for d in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def find_join_candidates(self, file_ids: list) -> list:
        """Suggest likely join keys across the given files by name match plus sampled value overlap."""
        for file_id in file_ids:
            self._check_assigned(file_id)

        candidates = []
        for i in range(len(file_ids)):
            for j in range(i + 1, len(file_ids)):
                file_a, file_b = file_ids[i], file_ids[j]
                table_a, table_b = self._table(file_a), self._table(file_b)
                cols_a = [d[0] for d in self.con.execute(f"SELECT * FROM {table_a} LIMIT 0").description]
                cols_b = [d[0] for d in self.con.execute(f"SELECT * FROM {table_b} LIMIT 0").description]

                for col_a in cols_a:
                    for col_b in cols_b:
                        if col_a.lower() != col_b.lower():
                            continue

                        set_a = {r[0] for r in self.con.execute(
                            f"SELECT DISTINCT {self._quote_ident(col_a)} FROM {table_a} LIMIT 1000").fetchall()}
                        set_b = {r[0] for r in self.con.execute(
                            f"SELECT DISTINCT {self._quote_ident(col_b)} FROM {table_b} LIMIT 1000").fetchall()}

                        if not set_a or not set_b:
                            continue

                        overlap = len(set_a & set_b) / len(set_a)
                        candidates.append({
                            "file_a": file_a,
                            "column_a": col_a,
                            "file_b": file_b,
                            "column_b": col_b,
                            "match_confidence": round(overlap, 2),
                        })
        return candidates

    def query_data(self, sql: str, file_ids: list, row_cap: int = 500, timeout_seconds: int = 15) -> QueryResult:
        """Run a SQL query against the given assigned files. Write the SQL using each file's
        table_name (from list_allowed_files/inspect_schema), never its file_id - file_id can
        contain characters (dots, hyphens) that are not valid unquoted SQL identifiers."""
        for file_id in file_ids:
            self._check_assigned(file_id)
        result = run_query(self.con, sql, row_cap, timeout_seconds)
        return QueryResult(**result)

    def export_query(self, sql: str, file_ids: list, name: str) -> dict:
        """Run a SQL query and persist the FULL result (not just a preview) as a new Parquet
        file, for when the objective needs the actual computed data to exist afterward - e.g.
        the user asked for this result as a CSV or dashboard, not just an answer in words.
        Use query_data instead when you only need to see the result yourself. name should be a
        short, descriptive label for what's in the result (e.g. "revenue_by_region"). Returns
        output_ref, row_count, and columns - report output_ref in your findings' artifact_refs
        so it can be exported later."""
        for file_id in file_ids:
            self._check_assigned(file_id)
        if self.storage is None:
            raise RuntimeError("no storage configured for this agent, cannot persist results")

        dataframe = self.con.execute(sql).df()
        safe_name = re.sub(r"[^0-9a-zA-Z_]", "_", name)[:60] or "result"
        result_id = f"{safe_name}_{uuid.uuid4().hex[:8]}"
        output_ref = self.storage.write(dataframe, f"{self.workspace_id}/{result_id}.parquet")

        return {
            "output_ref": output_ref,
            "row_count": len(dataframe),
            "columns": [str(c) for c in dataframe.columns],
        }

    def aggregate(self, file_ids: list, group_by: list, metrics: list[MetricSpec], filters: Optional[dict] = None) -> QueryResult:
        """Group-by + sum/avg/count/min/max convenience wrapper over query_data.
        Each item in metrics must have: column (str), op (one of sum|avg|count|min|max), alias (optional str).
        Each metric applies its op to the WHOLE column within each group - there is no per-value
        condition. For "count of X where column = A" vs "where column = B" within the same
        group (e.g. male vs female counts per job title), this tool cannot express that; use
        query_data/export_query with raw SQL (CASE WHEN ... END, or FILTER (WHERE ...)) instead."""
        for file_id in file_ids:
            self._check_assigned(file_id)

        select_parts = [self._quote_ident(col) for col in group_by]
        for raw_metric in metrics:
            metric = self._to_metric(raw_metric)
            alias = metric.alias or f"{metric.op}_{metric.column}"
            select_parts.append(
                f"{metric.op.upper()}({self._quote_ident(metric.column)}) AS {self._quote_ident(alias)}"
            )

        sql = f"SELECT {', '.join(select_parts)} FROM {self._table(file_ids[0])}"

        if filters:
            conditions = [f"{self._quote_ident(col)} = '{val}'" for col, val in filters.items()]
            sql += f" WHERE {' AND '.join(conditions)}"

        if group_by:
            sql += f" GROUP BY {', '.join(self._quote_ident(col) for col in group_by)}"

        result = run_query(self.con, sql, row_cap=500, timeout_seconds=15)
        return QueryResult(**result)

    @staticmethod
    def _to_metric(metric) -> MetricSpec:
        if isinstance(metric, MetricSpec):
            return metric
        op = metric.get("op") or metric.get("type")
        return MetricSpec(column=metric["column"], op=op, alias=metric.get("alias"))

    def describe_column(self, file_id: str, column: str) -> ColumnProfile:
        """Profile one column: min/max/mean, null count, distinct count, and top values.
        mean is only meaningful for numeric columns - it's None for text/date/other columns
        rather than erroring."""
        self._check_assigned(file_id)
        table = self._table(file_id)
        quoted_col = self._quote_ident(column)

        min_val, max_val, null_count, distinct_count = self.con.execute(
            f"SELECT MIN({quoted_col}), MAX({quoted_col}), "
            f"COUNT(*) FILTER (WHERE {quoted_col} IS NULL), COUNT(DISTINCT {quoted_col}) FROM {table}"
        ).fetchone()

        try:
            mean_val = self.con.execute(f"SELECT AVG({quoted_col}) FROM {table}").fetchone()[0]
        except Exception:
            mean_val = None

        top = self.con.execute(
            f"SELECT {quoted_col}, COUNT(*) AS c FROM {table} GROUP BY {quoted_col} ORDER BY c DESC LIMIT 10"
        ).fetchall()

        return ColumnProfile(
            min=min_val,
            max=max_val,
            mean=mean_val,
            null_count=null_count,
            distinct_count=distinct_count,
            top_values=[(r[0], r[1]) for r in top],
        )

    def validate_result(self, result: QueryResult, expected_shape: Optional[dict] = None) -> ValidationReport:
        """Sanity-check a query result: empty results, unexpected row counts, negative revenue-like values."""
        warnings = []

        if result.row_count == 0:
            warnings.append("query returned 0 rows")

        if expected_shape:
            min_rows = expected_shape.get("min_rows")
            if min_rows is not None and result.row_count < min_rows:
                warnings.append(f"expected at least {min_rows} rows, got {result.row_count}")

        for row in result.rows:
            for key, value in row.items():
                if isinstance(value, (int, float)) and value < 0 and "revenue" in key.lower():
                    warnings.append(f"negative value in revenue-like column '{key}'")
                    break

        return ValidationReport(passed=len(warnings) == 0, warnings=warnings)
