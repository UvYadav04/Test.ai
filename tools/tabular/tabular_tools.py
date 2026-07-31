import logging
import time
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

logger = logging.getLogger("tools.tabular")


class TabularTools:
    def __init__(
        self, assigned_files: list, storage=None, workspace_id: str = "default",
        chat_id: str = "default", sandbox_manager=None, reports_dir: str = "data/reports",
        chart_capacity_checker=None,
    ):
        self.assigned_files = {f.file_id: f for f in assigned_files}
        self.con = connect()
        self.storage = storage
        self.workspace_id = workspace_id
        self.chat_id = chat_id
        self.root_dir = getattr(storage, "root_dir", None)
        self.reporting = ReportingTools(storage, output_dir=reports_dir) if storage else None
        self.table_names = {}
        for file_ref in assigned_files:
            self.table_names[file_ref.file_id] = register_view(
                self.con, file_ref.file_id, self.workspace_id, self.root_dir
            )

        self._sandbox = (
            PythonSandbox(self.root_dir, session_id=self.chat_id, manager=sandbox_manager)
            if self.root_dir else None
        )
        self.saved_artifacts: dict = {}
        self.charts_created: list = []
       
        self.chart_capacity_checker = chart_capacity_checker

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
        """Execute Python code in an isolated sandbox. Pass every assigned file_id you need.

        Write ONE complete script that does the whole task (query, transform, compute) - do not
        split the task across multiple run_python calls to explore step by step. Only call this
        again if the actual output/error from this call forces a change you couldn't predict.

        Already defined in the global namespace - never import or redefine them:
        - dfs: {table_name: DataFrame} for every assigned file
        - describe(df): summary of a DataFrame
        - preview(df): small preview of a DataFrame
        - sql(query): runs SQL, returns a DataFrame
        - save(df, name): persists a DataFrame, returns its file_id

        Only pandas (`pd`) and duckdb are installed - no matplotlib, sklearn, or other
        third-party packages. Do not import anything; use the given functions directly."""
        if self._sandbox is None:
            raise RuntimeError("no storage configured for this agent, cannot run the sandbox")
        for file_id in file_ids:
            self._check_assigned(file_id)
        tables = {self._table(fid): fid for fid in file_ids}

        logger.info(
            "run_python: chat=%s workspace=%s file_ids=%s code_chars=%d",
            self.chat_id, self.workspace_id, file_ids, len(code),
        )
        t0 = time.perf_counter()
        try:
            result = self._sandbox.run(code, tables, self.workspace_id)
        except SandboxExecutionError as exc:
            logger.warning(
                "run_python: chat=%s raised after %.1fms: %s",
                self.chat_id, (time.perf_counter() - t0) * 1000, exc,
            )
            return {"stdout": "", "saved": [], "error": str(exc)}

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if result.get("error"):
            logger.warning(
                "run_python: chat=%s completed with error in %.1fms: %s",
                self.chat_id, elapsed_ms, result["error"],
            )
        else:
            logger.info(
                "run_python: chat=%s completed in %.1fms (saved=%d artifact(s))",
                self.chat_id, elapsed_ms, len(result.get("saved") or []),
            )

        for entry in result.get("saved") or []:
            file_id = entry.get("file_id")
            if file_id:
                self.saved_artifacts[file_id] = entry
        return result

    _VALID_CHART_TYPES = {
        "bar", "line", "timeline", "scatter3d", "surface", "pie", "histogram", "box", "heatmap",
    }

    async def create_visualizations(self, visualizations: list[dict]) -> dict:
        """Generate charts for one or more previously saved artifacts.

        Call this ONCE per run after all required save() calls. Pass every desired chart
        in a single `visualizations` list.

        Each visualization must include:
        - file_id: file_id returned by save() in this session.
        - chart_type: "bar", "line", "timeline", "scatter3d", "surface", "pie", "histogram",
          "box", or "heatmap".
        - title: Chart title.
        - Required column mappings based on chart_type:
            * bar/line:
                - label_column + value_columns (wide format), OR
                - label_column + series_column + value_column (long format)
            * timeline:
                - time_column + value_columns, OR
                - time_column + series_column + value_column
            * scatter3d/surface:
                - x_column, y_column, z_column
            * pie:
                - label_column (category/slice) + value_column (single numeric)
            * histogram:
                - value_column (single numeric column to bin/count)
                - optional series_column to overlay one histogram per group
                - optional bins (int) to control bucket count - auto if omitted
            * box:
                - value_column (single numeric column)
                - optional label_column to draw one box per category instead of a single box
            * heatmap:
                - x_column, y_column, z_column (z is averaged per x/y cell)

        Column names must exactly match those in the saved artifact.

        Each chart is validated independently. Invalid requests fail individually without
        affecting the others.

        Returns:
        {
            "status": "success" | "partial_error" | "error",
            "charts": [...],   # Successfully generated charts
            "errors": [...]    # Validation/generation failures
        }

        If the account's chart limit has been reached, no charts are generated. Do not
        retry; inform the user that the chart limit has been reached.
        """
        if self.reporting is None:
            return {
                "status": "error", "charts": [],
                "errors": [{"error": "no storage configured for this agent, cannot generate charts"}],
            }
        if self.chart_capacity_checker is not None and not await self.chart_capacity_checker():
            return {
                "status": "error", "charts": [],
                "errors": [{
                    "error": (
                        "Chart limit reached for this account - no more charts can be created. "
                        "Do not attempt to generate a chart; tell the user the limit has been hit."
                    ),
                }],
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
        bins = raw.get("bins")

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
            ("y_column", y_column), ("z_column", z_column), ("bins", bins),
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
        elif chart_type == "pie":
            if not (label_column and value_column):
                return (
                    "chart_type='pie' requires label_column (the slice category) and "
                    f"value_column (a single numeric column) - got label_column={label_column!r}, "
                    f"value_column={value_column!r}."
                )
        elif chart_type == "histogram":
            if not value_column:
                return (
                    "chart_type='histogram' requires value_column (the numeric column to bin) - "
                    "optionally series_column to overlay one histogram per group."
                )
        elif chart_type == "box":
            if not value_column:
                return (
                    "chart_type='box' requires value_column (the numeric column to summarize) - "
                    "optionally label_column to draw one box per category."
                )
        elif chart_type == "heatmap":
            if not (x_column and y_column and z_column):
                return (
                    "chart_type='heatmap' requires x_column, y_column, AND z_column all set - got "
                    f"x_column={x_column!r}, y_column={y_column!r}, z_column={z_column!r}."
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
