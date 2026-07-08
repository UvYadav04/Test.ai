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
    def __init__(self, assigned_files: list):
        self.assigned_files = {f.file_id: f for f in assigned_files}
        self.con = connect()
        for file_ref in assigned_files:
            register_view(self.con, file_ref.file_id, file_ref.output_ref)

    def _check_assigned(self, file_id: str) -> None:
        if file_id not in self.assigned_files:
            raise ValueError(f"file_id '{file_id}' is not assigned to this agent")

    def list_allowed_files(self) -> list:
        """Return metadata (row count, columns) for every file assigned to this agent."""
        files = []
        for file_id, file_ref in self.assigned_files.items():
            row_count = self.con.execute(f"SELECT COUNT(*) FROM {file_id}").fetchone()[0]
            columns = [d[0] for d in self.con.execute(f"SELECT * FROM {file_id} LIMIT 0").description]
            files.append({
                "file_id": file_id,
                "filename": file_ref.filename,
                "output_ref": file_ref.output_ref,
                "row_count": row_count,
                "columns": columns,
            })
        return files

    def inspect_schema(self, file_id: str) -> SchemaInfo:
        """Return column names, dtypes, nullability, and likely key columns for one assigned file."""
        self._check_assigned(file_id)

        info = self.con.execute(f"DESCRIBE SELECT * FROM {file_id}").fetchall()
        columns = [row[0] for row in info]
        dtypes = {row[0]: row[1] for row in info}
        nullable = {row[0]: row[2] == "YES" for row in info}

        total = self.con.execute(f"SELECT COUNT(*) FROM {file_id}").fetchone()[0]
        likely_keys = []
        for col in columns:
            if col.lower().endswith("_id"):
                likely_keys.append(col)
                continue
            distinct = self.con.execute(f"SELECT COUNT(DISTINCT {col}) FROM {file_id}").fetchone()[0]
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
        result = self.con.execute(f"SELECT * FROM {file_id} LIMIT {n}")
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
                cols_a = [d[0] for d in self.con.execute(f"SELECT * FROM {file_a} LIMIT 0").description]
                cols_b = [d[0] for d in self.con.execute(f"SELECT * FROM {file_b} LIMIT 0").description]

                for col_a in cols_a:
                    for col_b in cols_b:
                        if col_a.lower() != col_b.lower():
                            continue

                        set_a = {r[0] for r in self.con.execute(
                            f"SELECT DISTINCT {col_a} FROM {file_a} LIMIT 1000").fetchall()}
                        set_b = {r[0] for r in self.con.execute(
                            f"SELECT DISTINCT {col_b} FROM {file_b} LIMIT 1000").fetchall()}

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
        """Run a SQL query against the given assigned files (each file_id is a queryable table name)."""
        for file_id in file_ids:
            self._check_assigned(file_id)
        result = run_query(self.con, sql, row_cap, timeout_seconds)
        return QueryResult(**result)

    def aggregate(self, file_ids: list, group_by: list, metrics: list[MetricSpec], filters: dict = None) -> QueryResult:
        """Group-by + sum/avg/count/min/max convenience wrapper over query_data.
        Each item in metrics must have: column (str), op (one of sum|avg|count|min|max), alias (optional str)."""
        for file_id in file_ids:
            self._check_assigned(file_id)

        select_parts = list(group_by)
        for raw_metric in metrics:
            metric = self._to_metric(raw_metric)
            alias = metric.alias or f"{metric.op}_{metric.column}"
            select_parts.append(f"{metric.op.upper()}({metric.column}) AS {alias}")

        sql = f"SELECT {', '.join(select_parts)} FROM {file_ids[0]}"

        if filters:
            conditions = [f"{col} = '{val}'" for col, val in filters.items()]
            sql += f" WHERE {' AND '.join(conditions)}"

        if group_by:
            sql += f" GROUP BY {', '.join(group_by)}"

        result = run_query(self.con, sql, row_cap=500, timeout_seconds=15)
        return QueryResult(**result)

    @staticmethod
    def _to_metric(metric) -> MetricSpec:
        if isinstance(metric, MetricSpec):
            return metric
        op = metric.get("op") or metric.get("type")
        return MetricSpec(column=metric["column"], op=op, alias=metric.get("alias"))

    def describe_column(self, file_id: str, column: str) -> ColumnProfile:
        """Profile one column: min/max/mean, null count, distinct count, and top values."""
        self._check_assigned(file_id)

        row = self.con.execute(
            f"SELECT MIN({column}), MAX({column}), AVG({column}), "
            f"COUNT(*) FILTER (WHERE {column} IS NULL), COUNT(DISTINCT {column}) FROM {file_id}"
        ).fetchone()

        top = self.con.execute(
            f"SELECT {column}, COUNT(*) AS c FROM {file_id} GROUP BY {column} ORDER BY c DESC LIMIT 10"
        ).fetchall()

        return ColumnProfile(
            min=row[0],
            max=row[1],
            mean=row[2],
            null_count=row[3],
            distinct_count=row[4],
            top_values=[(r[0], r[1]) for r in top],
        )

    def validate_result(self, result: QueryResult, expected_shape: dict = None) -> ValidationReport:
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
