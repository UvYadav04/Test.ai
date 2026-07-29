from typing import Optional

from sandbox.path_resolver import new_artifact_id
from tools.reporting.models import ChartSpec
from tools.reporting.reporting_tools import ReportingTools
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
from tools.tabular.sandbox_executor import PythonSandbox, SandboxExecutionError


class TabularTools:
    def __init__(
        self, assigned_files: list, storage=None, workspace_id: str = "default",
        investigation_id: str = "default", sandbox_manager=None, reports_dir: str = "data/reports",
    ):
        self.assigned_files = {f.file_id: f for f in assigned_files}
        self.con = connect()
        self.storage = storage
        self.workspace_id = workspace_id
        self.investigation_id = investigation_id
        self.root_dir = getattr(storage, "root_dir", None)
        self.reporting = ReportingTools(storage, output_dir=reports_dir) if storage else None
        self.table_names = {}
        for file_ref in assigned_files:
            self.table_names[file_ref.file_id] = register_view(
                self.con, file_ref.file_id, self.workspace_id, self.root_dir
            )

        self._sandbox = (
            PythonSandbox(self.root_dir, investigation_id=self.investigation_id, manager=sandbox_manager)
            if self.root_dir else None
        )
        self.saved_artifacts: dict = {}
        self.charts_created: list = []

    def _check_assigned(self, file_id: str) -> None:
        if file_id not in self.assigned_files:
            raise ValueError(f"file_id '{file_id}' is not assigned to this agent")

    def _table(self, file_id: str) -> str:
        return self.table_names[file_id]

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    def list_allowed_files(self) -> list:
        """Return metadata (row count, columns, queryable table_name) for every file assigned to
        this agent. Use table_name (not file_id) to refer to a file inside run_python's dfs
        dict and sql() calls - file_id may contain characters (dots, hyphens) that aren't valid
        identifiers."""
        files = []
        for file_id, file_ref in self.assigned_files.items():
            table = self._table(file_id)
            row_count = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            columns = [d[0] for d in self.con.execute(f"SELECT * FROM {table} LIMIT 0").description]
            files.append({
                "file_id": file_id,
                "table_name": table,
                "filename": file_ref.filename,
                "row_count": row_count,
                "columns": columns,
            })
        return files

    def run_python(self, code: str, file_ids: list) -> dict:
        """
        Execute Python code in an isolated sandbox. Pass file ids to be used.

        Use table name for accessing the dataframe.

        Write only executable Python code.

        The execution environment already provides these globals.
        The following variables and functions are ALREADY DEFINED in the global namespace:

        Variables:
        - dfs

       Functions:
        - describe(df)
          Returns a summary of a DataFrame.

        - preview(df)
          Returns a preview of a DataFrame.

        - sql(query)
        Executes a SQL query and returns a pandas DataFrame.

        - save(df, name)
        Saves a DataFrame and returns its file_id.

        Never import them.
        Never redefine them.
        Never write `from ... import ...` or `import ... as ...`.

        Use them directly.

        Persist reusable outputs with save().

        AVAILABLE LIBRARIES - this sandbox only has these installed, nothing else:
        - pandas (as `pd`) - numpy comes along with it as pandas' own dependency, so basic
          numpy usage generally works, but it is not a supported/guaranteed part of this
          environment - prefer pandas/DuckDB SQL over numpy where you have a choice.
        - duckdb (as `duckdb`, wrapped by sql() above - you don't need to import or call it directly)

        Do NOT import any library or module, only use the given functions.
        """
        if self._sandbox is None:
            raise RuntimeError("no storage configured for this agent, cannot run the sandbox")
        for file_id in file_ids:
            self._check_assigned(file_id)
        tables = {self._table(fid): fid for fid in file_ids}
        try:
            result = self._sandbox.run(code, tables, self.workspace_id)
        except SandboxExecutionError as exc:
            return {"stdout": "", "saved": [], "error": str(exc)}

        for entry in result.get("saved") or []:
            file_id = entry.get("file_id")
            if file_id:
                self.saved_artifacts[file_id] = entry
        return result

    _VALID_CHART_TYPES = {"bar", "line", "timeline", "scatter3d", "surface"}

    def create_visualizations(self, visualizations: list[dict]) -> dict:
        """Validate, generate, and save charts for one or more artifacts you already save()'d -
        in a SINGLE call. Pass every chart the objective needs at once (one dict per chart in
        `visualizations`) rather than calling this more than once per run; if you already know
        every chart you need before calling this, save every artifact first, then make one
        create_visualizations call with all of them together.

        You are the only one who knows how each artifact should be charted: you read the
        objective, planned the query, and interpreted the result - nobody downstream ever sees
        the raw data, so if you don't say which column is the category and which is the metric,
        nobody else can work it out correctly afterward.

        Each item in `visualizations` is a dict with:
        - file_id (required): a real file_id this session's save() already returned - call
          list_allowed_files/run_python first, never a guessed or invented id.
        - chart_type (required): one of "bar", "line", "timeline", "scatter3d", "surface".
          - "bar"/"line": label_column + value_columns for a single grouping column with one or
            more numeric series; or label_column + series_column + value_column when the result
            has TWO grouping columns and one metric (e.g. columns Region, Gender, AverageAge ->
            label_column="Region", series_column="Gender", value_column="AverageAge").
          - "timeline": time_column is required, plus either value_columns (wide: one series per
            column) or series_column + value_column (long/tidy: one series per distinct value in
            series_column).
          - "scatter3d"/"surface": x_column, y_column, and z_column are all required.
        - title (required): a short, specific chart title (e.g. "Average Age per Region for
          Exited Customers") - this is what gets shown verbatim, not the artifact's file_id.
        - label_column/value_columns/time_column/series_column/value_column/x_column/y_column/
          z_column (optional, chart_type-dependent, see above): must be EXACT column names from
          that artifact (as returned by save() / seen in this session's run_python results) -
          never a renamed, abbreviated, or invented name.

        Every visualization request is validated and generated INDEPENDENTLY - one bad request
        (an unsupported chart_type, a file_id never save()'d this session, a wrong column name, a
        missing required column for that chart_type) returns an error for THAT chart only and
        never prevents the others in the same call from being generated. This never raises, so a
        mistake in one entry doesn't end your run early.

        Returns {"status": "success"|"partial_error"|"error", "charts": [...], "errors": [...]}.
        Each successful entry in "charts" has artifact_file_id, chart_id, chart_type, title, and
        location (the chart's saved file) - it's already generated and will be attached to the
        final answer automatically, you don't need to do anything else with it, just mention it
        in your summary. Each entry in "errors" has the request's index, requested_file_id (if
        available), and an "error" message explaining exactly what to fix."""
        if self.reporting is None:
            return {
                "status": "error", "charts": [],
                "errors": [{"error": "no storage configured for this agent, cannot generate charts"}],
            }

        charts = []
        errors = []
        for index, raw in enumerate(visualizations or []):
            requested_file_id = raw.get("file_id") if isinstance(raw, dict) else None
            try:
                spec_kwargs, error = self._validate_chart_request(raw)
            except Exception as exc:
                spec_kwargs, error = None, f"invalid visualization request: {exc}"

            if error:
                errors.append({"index": index, "requested_file_id": requested_file_id, "error": error})
                continue

            try:
                chart_id = new_artifact_id(f"chart_{spec_kwargs['chart_type']}")
                folder_name = f"{chart_id}_{spec_kwargs['title'][:30]}"
                location = self.reporting.render_single_chart(
                    self.workspace_id, ChartSpec(**spec_kwargs), name=folder_name,
                )
            except Exception as exc:
                errors.append({
                    "index": index, "requested_file_id": requested_file_id,
                    "error": f"chart generation failed: {exc}",
                })
                continue

            chart_entry = {
                "artifact_file_id": spec_kwargs["file_id"],
                "chart_id": chart_id,
                "chart_type": spec_kwargs["chart_type"],
                "title": spec_kwargs["title"],
                "location": location,
            }
            self.charts_created.append(chart_entry)
            charts.append(chart_entry)

        if not visualizations:
            status = "error"
            errors = errors or [{"error": "visualizations must be a non-empty list of chart requests"}]
        elif charts and not errors:
            status = "success"
        elif charts and errors:
            status = "partial_error"
        else:
            status = "error"

        return {"status": status, "charts": charts, "errors": errors}

    def _validate_chart_request(self, raw) -> tuple:
        if not isinstance(raw, dict):
            return None, f"each visualization request must be an object/dict, got {type(raw).__name__}"

        file_id = raw.get("file_id")
        chart_type = raw.get("chart_type")
        title = raw.get("title")
        label_column = raw.get("label_column")
        value_columns = raw.get("value_columns")
        time_column = raw.get("time_column")
        series_column = raw.get("series_column")
        value_column = raw.get("value_column")
        x_column = raw.get("x_column")
        y_column = raw.get("y_column")
        z_column = raw.get("z_column")

        if not file_id:
            return None, "file_id is required"
        if not title:
            return None, "title is required"
        if chart_type not in self._VALID_CHART_TYPES:
            return None, (
                f"chart_type {chart_type!r} is not supported - choose one of "
                f"{sorted(self._VALID_CHART_TYPES)}."
            )

        entry = self.saved_artifacts.get(file_id)
        if entry is None:
            return None, (
                f"file_id {file_id!r} was not save()'d by any run_python call yet in this "
                f"session. Call save(df, name) inside run_python first, then pass the exact "
                f"file_id it returned. Artifacts saved so far: {sorted(self.saved_artifacts.keys())}"
            )

        columns = entry.get("columns") or []
        named_single = {
            "label_column": label_column, "time_column": time_column,
            "series_column": series_column, "value_column": value_column,
            "x_column": x_column, "y_column": y_column, "z_column": z_column,
        }
        bad = {arg: val for arg, val in named_single.items() if val and val not in columns}
        for val in value_columns or []:
            if val not in columns:
                bad[f"value_columns={val!r}"] = val
        if bad:
            return None, (
                f"column(s) not found in artifact {file_id!r}: {bad} - the REAL columns in this "
                f"artifact are exactly: {columns}. Use one of these verbatim."
            )

        missing = self._missing_chart_columns(
            chart_type, label_column, value_columns, time_column, series_column, value_column,
            x_column, y_column, z_column,
        )
        if missing:
            return None, missing

        spec_kwargs = {"file_id": file_id, "chart_type": chart_type, "title": title}
        for key, val in (
            ("label_column", label_column), ("value_columns", value_columns),
            ("time_column", time_column), ("series_column", series_column),
            ("value_column", value_column), ("x_column", x_column),
            ("y_column", y_column), ("z_column", z_column),
        ):
            if val:
                spec_kwargs[key] = val

        return spec_kwargs, None

    @staticmethod
    def _missing_chart_columns(
        chart_type, label_column, value_columns, time_column, series_column, value_column,
        x_column, y_column, z_column,
    ) -> Optional[str]:
        if chart_type in ("bar", "line"):
            wide = bool(label_column and value_columns)
            tidy = bool(label_column and series_column and value_column)
            if not (wide or tidy):
                return (
                    f"chart_type={chart_type!r} needs EITHER (label_column + value_columns) for "
                    f"a single grouping column, OR (label_column + series_column + value_column) "
                    f"for two grouping columns - got label_column={label_column!r}, "
                    f"value_columns={value_columns!r}, series_column={series_column!r}, "
                    f"value_column={value_column!r}."
                )
        elif chart_type == "timeline":
            if not time_column:
                return "chart_type='timeline' requires time_column to be set."
            wide = bool(value_columns)
            tidy = bool(series_column and value_column)
            if not (wide or tidy):
                return (
                    "chart_type='timeline' needs time_column plus EITHER value_columns (one "
                    "series per column) OR (series_column + value_column) (one series per "
                    "distinct series_column value)."
                )
        elif chart_type in ("scatter3d", "surface"):
            if not (x_column and y_column and z_column):
                return (
                    f"chart_type={chart_type!r} requires x_column, y_column, AND z_column all "
                    f"set - got x_column={x_column!r}, y_column={y_column!r}, z_column={z_column!r}."
                )
        return None

    def inspect_schema(self, file_id: str) -> SchemaInfo:
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

    def sample_rows(self, file_id: str, n: int = 5) -> list:
        self._check_assigned(file_id)
        n = min(n, 50)
        table = self._table(file_id)
        result = self.con.execute(f"SELECT * FROM {table} LIMIT {n}")
        columns = [d[0] for d in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def find_join_candidates(self, file_ids: list) -> list:
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

    def query_data(
        self,
        sql: str,
        file_ids: list,
        persist: bool = False,
        name: Optional[str] = None,
        preview_rows: int = 10,
        timeout_seconds: int = 15,
    ) -> QueryResult:
        for file_id in file_ids:
            self._check_assigned(file_id)
        preview_rows = max(1, min(preview_rows, 20))

        if persist:
            if self.storage is None:
                raise RuntimeError("no storage configured for this agent, cannot persist results")
            clean_sql = sql.strip()
            if clean_sql.endswith(";"):
                clean_sql = clean_sql[:-1].rstrip()

            dataframe = self.con.execute(clean_sql).df()
            file_id = new_artifact_id(name or "result")
            self.storage.write(dataframe, f"{self.workspace_id}/{file_id}.parquet")

            row_count = len(dataframe)
            preview = dataframe.head(preview_rows).to_dict(orient="records")
            return QueryResult(
                columns=[str(c) for c in dataframe.columns],
                rows=preview,
                row_count=row_count,
                truncated=row_count > len(preview),
                error=None,
                file_id=file_id,
            )

        result = run_query(self.con, sql, preview_rows, timeout_seconds)
        return QueryResult(**result, file_id=None)

    def aggregate(
        self,
        file_ids: list,
        group_by: list,
        metrics: list[MetricSpec],
        filters: Optional[dict] = None,
        name: Optional[str] = None,
    ) -> QueryResult:
        for file_id in file_ids:
            self._check_assigned(file_id)
        if self.storage is None:
            raise RuntimeError("no storage configured for this agent, cannot persist results")

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

        dataframe = self.con.execute(sql).df()
        file_id = new_artifact_id(name or "aggregate")
        self.storage.write(dataframe, f"{self.workspace_id}/{file_id}.parquet")

        row_count = len(dataframe)
        preview = dataframe.head(20).to_dict(orient="records")
        return QueryResult(
            columns=[str(c) for c in dataframe.columns],
            rows=preview,
            row_count=row_count,
            truncated=row_count > len(preview),
            error=None,
            file_id=file_id,
        )

    @staticmethod
    def _to_metric(metric) -> MetricSpec:
        if isinstance(metric, MetricSpec):
            return metric
        op = metric.get("op") or metric.get("type")
        return MetricSpec(column=metric["column"], op=op, alias=metric.get("alias"))

    def describe_column(self, file_id: str, column: str) -> ColumnProfile:
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
