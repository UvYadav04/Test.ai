import json
import math
import os
import re
import shutil
import uuid
from datetime import date
from typing import Optional

import pandas as pd

from tools.reporting.models import ChartSpec


class ReportingTools:
    def __init__(self, storage, output_dir: str = "data/reports"):
        self.storage = storage
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_csv(self, output_ref: str, name: Optional[str] = None) -> str:
        """Convert an existing data artifact (an output_ref from a table_ref or from a
        persisted query_data call) into a CSV file. Creates a new dated folder (today's date + name) and
        writes the CSV there, alongside a copy of the source data file, so the request's
        output is self-contained. Returns the CSV file path."""
        folder = self._new_folder(name, "export")
        dataframe = self.storage.read(output_ref)
        path = os.path.join(folder, "data.csv")
        dataframe.to_csv(path, index=False)
        self._copy_source(output_ref, folder)
        return path

    def generate_markdown_report(
        self,
        title: str,
        objective: str,
        summary: str,
        findings: list,
        open_questions: Optional[list] = None,
        name: Optional[str] = None,
    ) -> str:
        """Build a markdown report file from your synthesized investigation results. Call this
        when the user wants a written report file, not just a chat answer - pass your own
        summary and findings text (as a list of short strings), not raw tool output. Creates a
        new dated folder (today's date + name, falling back to a slug of title) and writes the
        report there."""
        lines = [f"# {title}", "", f"**Objective:** {objective}", "", "## Summary", "", summary]

        if findings:
            lines += ["", "## Findings"]
            for i, finding in enumerate(findings, 1):
                lines.append(f"{i}. {finding}")

        if open_questions:
            lines += ["", "## Open Questions"]
            for question in open_questions:
                lines.append(f"- {question}")

        folder = self._new_folder(name or title, "report")
        path = os.path.join(folder, "report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    def generate_dashboard(self, title: str, sections: list[ChartSpec], name: Optional[str] = None) -> str:
        """Build a single self-contained HTML dashboard from one or more existing data
        artifacts. Use this when the user wants a visual dashboard, not a CSV or written report.

        Each item in `sections` is a ChartSpec: {output_ref, chart_type, ...column names...}.
        You never pass or see actual data values here - only an output_ref (a file path) and
        column names you already know from query_data's returned `columns` list. The real
        numbers are read straight from the parquet file when the dashboard is built.

        chart_type options and which column names each needs:
        - "bar" / "line" (2D, Chart.js): EITHER label_column + value_columns (1+ numeric series
          - if omitted, the first non-numeric column and up to 5 numeric columns are used
          automatically) OR, for data with two grouping columns and one metric, label_column +
          series_column + value_column (one bar/line per distinct series_column value, grouped
          along label_column).
        - "timeline" (line chart over time, Chart.js): time_column (required) plus EITHER
          value_columns (wide data - one series per column) OR series_column + value_column
          (long/tidy data - one series per distinct value in series_column, e.g. columns date,
          job_title, count -> series_column="job_title", value_column="count").
        - "scatter3d" / "surface" (3 dimensions, Plotly): x_column, y_column, z_column (all
          required). "surface" needs every (x, y) combination present in the data to build a
          valid grid - use "scatter3d" instead if that can't be guaranteed.

        Creates a new dated folder (today's date + name, falling back to a slug of title) and
        writes the dashboard there, alongside copies of every source data file that fed it."""
        folder = self._new_folder(name or title, "dashboard")
        rendered_sections = []
        for raw_spec in sections:
            spec = self._to_chart_spec(raw_spec)
            dataframe = self.storage.read(spec.output_ref)
            self._copy_source(spec.output_ref, folder)

            if spec.chart_type in ("bar", "line"):
                rendered_sections.append(self._categorical_section(dataframe, spec))
            elif spec.chart_type == "timeline":
                rendered_sections.append(self._timeline_section(dataframe, spec))
            elif spec.chart_type in ("scatter3d", "surface"):
                rendered_sections.append(self._chart3d_section(dataframe, spec))
            else:
                raise ValueError(
                    f"unknown chart_type '{spec.chart_type}' - use one of: "
                    "bar, line, timeline, scatter3d, surface"
                )

        html = self._render_html(title, rendered_sections)
        path = os.path.join(folder, "dashboard.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    @staticmethod
    def _to_chart_spec(raw) -> ChartSpec:
        if isinstance(raw, ChartSpec):
            return raw
        if isinstance(raw, str):
            # backward-compatible shorthand: a bare output_ref string means an auto bar chart
            return ChartSpec(output_ref=raw)
        return ChartSpec(**raw)

    def _new_folder(self, name: Optional[str], default_stem: str) -> str:
        safe_name = re.sub(r"[^0-9a-zA-Z_-]", "_", name)[:60] if name else ""
        safe_name = safe_name.strip("_") or f"{default_stem}_{uuid.uuid4().hex[:8]}"
        folder = os.path.join(self.output_dir, date.today().isoformat(), safe_name)
        os.makedirs(folder, exist_ok=True)
        return folder

    @staticmethod
    def _copy_source(output_ref: str, folder: str) -> None:
        try:
            if os.path.isfile(output_ref):
                shutil.copy2(output_ref, folder)
        except OSError:
            pass

    @staticmethod
    def _safe(value):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    @classmethod
    def _pivot_datasets(cls, df: pd.DataFrame, index_col: str, series_col: str, value_col: str) -> tuple:
        """Shared long/tidy -> chart-ready pivot: one row per (index_col, series_col, value_col)
        becomes one label per distinct index_col value and one dataset per distinct series_col
        value. Used for grouped bar/line charts and for timeline charts with multiple series."""
        pivot = df.pivot_table(index=index_col, columns=series_col, values=value_col, aggfunc="sum")
        labels = [str(v) for v in pivot.index.tolist()]
        datasets = [
            {"label": str(col), "data": [cls._safe(v) for v in pivot[col].tolist()]}
            for col in pivot.columns
        ]
        return labels, datasets

    @classmethod
    def _categorical_section(cls, dataframe: pd.DataFrame, spec: ChartSpec) -> dict:
        if spec.series_column and spec.value_column:
            if not spec.label_column:
                raise ValueError(
                    f"chart_type '{spec.chart_type}' with series_column requires label_column "
                    "too (the categorical axis to group by)"
                )
            labels, datasets = cls._pivot_datasets(
                dataframe, spec.label_column, spec.series_column, spec.value_column
            )
        else:
            numeric_cols = [c for c in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[c])]
            label_col = spec.label_column or next(
                (c for c in dataframe.columns if c not in numeric_cols), dataframe.columns[0]
            )
            value_cols = spec.value_columns or numeric_cols[:5]
            rows = dataframe.head(50)
            labels = rows[label_col].astype(str).tolist()
            datasets = [
                {"label": col, "data": [cls._safe(v) for v in rows[col].tolist()]}
                for col in value_cols
            ]

        return {
            "kind": "chartjs",
            "chart_type": spec.chart_type,
            "title": spec.title or spec.output_ref,
            "labels": labels,
            "datasets": datasets,
        }

    @classmethod
    def _timeline_section(cls, dataframe: pd.DataFrame, spec: ChartSpec) -> dict:
        if not spec.time_column:
            raise ValueError("chart_type 'timeline' requires time_column")
        df = dataframe.sort_values(spec.time_column)

        if spec.series_column and spec.value_column:
            labels, datasets = cls._pivot_datasets(df, spec.time_column, spec.series_column, spec.value_column)
        else:
            value_cols = spec.value_columns or [
                c for c in df.columns
                if c != spec.time_column and pd.api.types.is_numeric_dtype(df[c])
            ][:5]
            if not value_cols:
                raise ValueError(
                    "chart_type 'timeline' requires either value_columns, or "
                    "series_column + value_column"
                )
            labels = df[spec.time_column].astype(str).tolist()
            datasets = [
                {"label": col, "data": [cls._safe(v) for v in df[col].tolist()]}
                for col in value_cols
            ]

        return {
            "kind": "chartjs",
            "chart_type": "line",
            "title": spec.title or spec.output_ref,
            "labels": labels,
            "datasets": datasets,
        }

    @classmethod
    def _chart3d_section(cls, dataframe: pd.DataFrame, spec: ChartSpec) -> dict:
        if not (spec.x_column and spec.y_column and spec.z_column):
            raise ValueError(f"chart_type '{spec.chart_type}' requires x_column, y_column, and z_column")
        df = dataframe.head(2000)

        if spec.chart_type == "surface":
            pivot = df.pivot_table(
                index=spec.y_column, columns=spec.x_column, values=spec.z_column, aggfunc="mean"
            )
            z_matrix = [[cls._safe(v) for v in row] for row in pivot.values.tolist()]
            return {
                "kind": "plotly",
                "plot_type": "surface",
                "title": spec.title or spec.output_ref,
                "x": [str(c) for c in pivot.columns.tolist()],
                "y": [str(i) for i in pivot.index.tolist()],
                "z": z_matrix,
            }

        return {
            "kind": "plotly",
            "plot_type": "scatter3d",
            "title": spec.title or spec.output_ref,
            "x": [cls._safe(v) for v in df[spec.x_column].tolist()],
            "y": [cls._safe(v) for v in df[spec.y_column].tolist()],
            "z": [cls._safe(v) for v in df[spec.z_column].tolist()],
        }

    @staticmethod
    def _render_html(title: str, sections: list) -> str:
        chartjs_sections = [s for s in sections if s["kind"] == "chartjs"]
        plotly_sections = [s for s in sections if s["kind"] == "plotly"]

        canvases = "\n".join(f'<canvas id="chart{i}"></canvas>' for i in range(len(chartjs_sections)))
        chart_scripts = "\n".join(
            f"""
            new Chart(document.getElementById('chart{i}'), {{
                type: {json.dumps(section["chart_type"])},
                data: {{
                    labels: {json.dumps(section["labels"])},
                    datasets: {json.dumps(section["datasets"])}
                }},
                options: {{ responsive: true, plugins: {{ title: {{ display: true, text: {json.dumps(section["title"])} }} }} }}
            }});
            """
            for i, section in enumerate(chartjs_sections)
        )

        plot_divs = "\n".join(
            f'<div id="plot{i}" style="width:100%;height:500px;margin-bottom:50px;"></div>'
            for i in range(len(plotly_sections))
        )
        plot_scripts = "\n".join(
            ReportingTools._plotly_script(i, section) for i, section in enumerate(plotly_sections)
        )

        plotly_cdn = (
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>'
            if plotly_sections else ""
        )

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
{plotly_cdn}
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; }}
canvas {{ margin-bottom: 50px; }}
</style>
</head>
<body>
<h1>{title}</h1>
{canvases}
{plot_divs}
<script>
{chart_scripts}
{plot_scripts}
</script>
</body>
</html>
"""

    @staticmethod
    def _plotly_script(i: int, section: dict) -> str:
        if section["plot_type"] == "surface":
            trace = {"type": "surface", "x": section["x"], "y": section["y"], "z": section["z"]}
        else:
            trace = {
                "type": "scatter3d",
                "mode": "markers",
                "x": section["x"],
                "y": section["y"],
                "z": section["z"],
            }
        layout = {"title": section["title"], "autosize": True, "margin": {"l": 0, "r": 0, "b": 0, "t": 40}}
        return f"Plotly.newPlot('plot{i}', [{json.dumps(trace)}], {json.dumps(layout)});"
