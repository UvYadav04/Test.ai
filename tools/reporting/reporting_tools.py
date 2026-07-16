import json
import math
import os
import re
import shutil
import uuid
from datetime import date, datetime
from typing import Optional

import pandas as pd

from tools.reporting.models import ChartSpec

# Warm terracotta-led palette matching the app's own theme - primary accent first, then a mix
# of warm and a few cool contrast tones so multi-series charts stay readable (not all one hue).
PALETTE = [
    "#CC785C",  # terracotta (primary accent)
    "#4A7C7C",  # muted teal (contrast)
    "#D9A566",  # warm gold
    "#8B6F9E",  # muted plum (contrast)
    "#B0562B",  # deep rust
    "#6B8F71",  # sage green (contrast)
    "#C4956C",  # warm tan
    "#5F7A99",  # dusty blue (contrast)
    "#A8754A",  # caramel brown
    "#D97757",  # coral orange
]

_HEX_SUFFIX_RE = re.compile(r"_[0-9a-f]{8}$")


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
        """Build a single self-contained, polished, interactive HTML dashboard from one or more
        existing data artifacts. Use this when the user wants a visual dashboard, not a CSV or
        written report.

        Each item in `sections` is a ChartSpec: {output_ref, chart_type, ...column names...}.
        You never pass or see actual data values here - only an output_ref (a file path) and
        column names you already know from a Tabular Agent's findings. The real numbers are
        read straight from the parquet file when the dashboard is built; charts are laid out
        automatically in a responsive grid regardless of how many sections you pass.

        chart_type options and which column names each needs:
        - "bar" / "line" (2D): EITHER label_column + value_columns (1+ numeric series - if
          omitted, the first non-numeric column and up to 5 numeric columns are used
          automatically) OR, for data with two grouping columns and one metric, label_column +
          series_column + value_column (one bar/line per distinct series_column value, grouped
          along label_column).
        - "timeline" (line chart over time): time_column (required) plus EITHER value_columns
          (wide data - one series per column) OR series_column + value_column (long/tidy data -
          one series per distinct value in series_column).
        - "scatter3d" / "surface" (3 dimensions): x_column, y_column, z_column (all required).
          "surface" needs every (x, y) combination present in the data to build a valid grid -
          use "scatter3d" instead if that can't be guaranteed.

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

        html = self._render_html(title, rendered_sections, source_count=len(sections))
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

    @staticmethod
    def _humanize(output_ref: str, fallback: str) -> str:
        base = os.path.splitext(os.path.basename(str(output_ref)))[0]
        base = _HEX_SUFFIX_RE.sub("", base)
        base = re.sub(r"[_-]+", " ", base).strip()
        return base.title() if base else fallback

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
        y_label = None
        if spec.series_column and spec.value_column:
            if not spec.label_column:
                raise ValueError(
                    f"chart_type '{spec.chart_type}' with series_column requires label_column "
                    "too (the categorical axis to group by)"
                )
            labels, datasets = cls._pivot_datasets(
                dataframe, spec.label_column, spec.series_column, spec.value_column
            )
            x_label = spec.label_column
            y_label = spec.value_column
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
            x_label = label_col
            y_label = value_cols[0] if len(value_cols) == 1 else None

        return {
            "kind": "chartjs",
            "chart_type": spec.chart_type,
            "title": spec.title or cls._humanize(spec.output_ref, "Chart"),
            "source": os.path.basename(str(spec.output_ref)),
            "row_count": len(dataframe),
            "x_label": x_label,
            "y_label": y_label,
            "labels": labels,
            "datasets": datasets,
        }

    @classmethod
    def _timeline_section(cls, dataframe: pd.DataFrame, spec: ChartSpec) -> dict:
        if not spec.time_column:
            raise ValueError("chart_type 'timeline' requires time_column")
        df = dataframe.sort_values(spec.time_column)
        y_label = None

        if spec.series_column and spec.value_column:
            labels, datasets = cls._pivot_datasets(df, spec.time_column, spec.series_column, spec.value_column)
            y_label = spec.value_column
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
            y_label = value_cols[0] if len(value_cols) == 1 else None

        return {
            "kind": "chartjs",
            "chart_type": "line",
            "title": spec.title or cls._humanize(spec.output_ref, "Timeline"),
            "source": os.path.basename(str(spec.output_ref)),
            "row_count": len(df),
            "x_label": spec.time_column,
            "y_label": y_label,
            "labels": labels,
            "datasets": datasets,
        }

    @classmethod
    def _chart3d_section(cls, dataframe: pd.DataFrame, spec: ChartSpec) -> dict:
        if not (spec.x_column and spec.y_column and spec.z_column):
            raise ValueError(f"chart_type '{spec.chart_type}' requires x_column, y_column, and z_column")
        df = dataframe.head(2000)
        title = spec.title or cls._humanize(spec.output_ref, "3D Chart")
        source = os.path.basename(str(spec.output_ref))

        if spec.chart_type == "surface":
            pivot = df.pivot_table(
                index=spec.y_column, columns=spec.x_column, values=spec.z_column, aggfunc="mean"
            )
            z_matrix = [[cls._safe(v) for v in row] for row in pivot.values.tolist()]
            return {
                "kind": "plotly",
                "plot_type": "surface",
                "title": title,
                "source": source,
                "row_count": len(df),
                "x_label": spec.x_column,
                "y_label": spec.y_column,
                "z_label": spec.z_column,
                "x": [str(c) for c in pivot.columns.tolist()],
                "y": [str(i) for i in pivot.index.tolist()],
                "z": z_matrix,
            }

        return {
            "kind": "plotly",
            "plot_type": "scatter3d",
            "title": title,
            "source": source,
            "row_count": len(df),
            "x_label": spec.x_column,
            "y_label": spec.y_column,
            "z_label": spec.z_column,
            "x": [cls._safe(v) for v in df[spec.x_column].tolist()],
            "y": [cls._safe(v) for v in df[spec.y_column].tolist()],
            "z": [cls._safe(v) for v in df[spec.z_column].tolist()],
        }

    # ------------------------------------------------------------------ rendering

    @staticmethod
    def _chip(chart_type: str) -> str:
        return {
            "bar": "Bar", "line": "Line", "timeline": "Timeline",
            "scatter3d": "3D Scatter", "surface": "3D Surface",
        }.get(chart_type, chart_type.title())

    @classmethod
    def _render_html(cls, title: str, sections: list, source_count: int) -> str:
        chartjs_sections = [s for s in sections if s["kind"] == "chartjs"]
        plotly_sections = [s for s in sections if s["kind"] == "plotly"]

        cards = []
        chart_scripts = []
        plot_scripts = []
        chartjs_i = 0
        plotly_i = 0

        for section in sections:
            chip = cls._chip(section.get("chart_type") or section.get("plot_type"))
            card_id = f"card{len(cards)}"
            if section["kind"] == "chartjs":
                canvas_id = f"chart{chartjs_i}"
                cards.append(cls._card_html(section, chip, f'<canvas id="{canvas_id}"></canvas>'))
                chart_scripts.append(cls._chartjs_script(canvas_id, section, chartjs_i))
                chartjs_i += 1
            else:
                plot_id = f"plot{plotly_i}"
                cards.append(cls._card_html(section, chip, f'<div id="{plot_id}" class="plot-el"></div>'))
                plot_scripts.append(cls._plotly_script(plot_id, section))
                plotly_i += 1

        plotly_cdn = (
            '<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>'
            if plotly_sections else ""
        )
        generated = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        file_word = "file" if source_count == 1 else "files"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
{plotly_cdn}
<style>{cls._css()}</style>
</head>
<body>
<div class="page">
  <header class="page-header">
    <div>
      <h1>{title}</h1>
      <p class="subtitle">Generated {generated} &middot; {source_count} data {file_word}</p>
    </div>
  </header>
  <main class="dashboard-grid">
    {"".join(cards)}
  </main>
  <footer class="page-footer">Built automatically from your data &mdash; hover any chart to explore.</footer>
</div>
<script>
const PALETTE = {json.dumps(PALETTE)};
function withAlpha(hex, alpha) {{
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}
Chart.defaults.font.family = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif";
Chart.defaults.font.size = 12.5;
Chart.defaults.color = "#83807A";
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.boxWidth = 8;
Chart.defaults.plugins.legend.labels.boxHeight = 8;
Chart.defaults.plugins.tooltip.backgroundColor = "#262624";
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.titleFont = {{ weight: '600' }};
{"".join(chart_scripts)}
{"".join(plot_scripts)}
</script>
</body>
</html>
"""

    @staticmethod
    def _card_html(section: dict, chip: str, body_html: str) -> str:
        row_count = section.get("row_count")
        stat = f'<span class="card-stat">{row_count:,} rows</span>' if row_count is not None else ""
        source = section.get("source", "")
        return f"""
    <section class="chart-card">
      <div class="card-header">
        <div>
          <h2>{section["title"]}</h2>
          <p class="card-source">{source}</p>
        </div>
        <div class="card-meta">
          <span class="chip">{chip}</span>
          {stat}
        </div>
      </div>
      <div class="chart-body">{body_html}</div>
    </section>"""

    @staticmethod
    def _chartjs_script(canvas_id: str, section: dict, index: int) -> str:
        chart_type = section["chart_type"]
        datasets = section["datasets"]
        multi = len(datasets) > 1
        datasets_json = json.dumps(datasets)

        dataset_styling = f"""
        opts.data.datasets.forEach(function(ds, i) {{
          const color = PALETTE[i % PALETTE.length];
          if ("{chart_type}" === "bar") {{
            ds.backgroundColor = withAlpha(color, 0.82);
            ds.hoverBackgroundColor = color;
            ds.borderRadius = 6;
            ds.borderSkipped = false;
            ds.maxBarThickness = 46;
          }} else {{
            ds.borderColor = color;
            ds.backgroundColor = withAlpha(color, 0.12);
            ds.pointBackgroundColor = color;
            ds.pointRadius = {"2.5" if multi else "3"};
            ds.pointHoverRadius = 5;
            ds.borderWidth = 2.5;
            ds.tension = 0.35;
            ds.fill = {"false" if multi else "true"};
          }}
        }});
        """

        x_label = section.get("x_label")
        y_label = section.get("y_label")
        x_title = f'{{ display: true, text: {json.dumps(x_label)}, color: "#A39E92", font: {{ weight: "600" }} }}' if x_label else '{ display: false }'
        y_title = f'{{ display: true, text: {json.dumps(y_label)}, color: "#A39E92", font: {{ weight: "600" }} }}' if y_label else '{ display: false }'

        return f"""
(function() {{
  const opts = {{
    type: {json.dumps(chart_type)},
    data: {{ labels: {json.dumps(section["labels"])}, datasets: {datasets_json} }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ display: {json.dumps(multi)}, position: 'bottom' }},
      }},
      scales: {{
        x: {{ title: {x_title}, grid: {{ display: false }}, ticks: {{ maxRotation: 40, minRotation: 0 }} }},
        y: {{ title: {y_title}, grid: {{ color: "#EDEAE0" }}, beginAtZero: true }}
      }}
    }}
  }};
  {dataset_styling}
  new Chart(document.getElementById({json.dumps(canvas_id)}), opts);
}})();
"""

    @staticmethod
    def _plotly_script(plot_id: str, section: dict) -> str:
        common_layout = {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif", "color": "#3D3929"},
            "margin": {"l": 10, "r": 10, "b": 10, "t": 10},
            "scene": {
                "xaxis": {"title": section.get("x_label") or "X", "gridcolor": "#EDEAE0"},
                "yaxis": {"title": section.get("y_label") or "Y", "gridcolor": "#EDEAE0"},
                "zaxis": {"title": section.get("z_label") or "Z", "gridcolor": "#EDEAE0"},
            },
        }
        # Warm terracotta-to-cream colorscale so 3D charts match the app's theme instead of
        # Plotly's default cool Viridis (blue-green-yellow).
        warm_colorscale = [
            [0.0, "#F5F4EE"], [0.25, "#E8C4A8"], [0.5, "#D9A566"],
            [0.75, "#CC785C"], [1.0, "#8B4A32"],
        ]
        if section["plot_type"] == "surface":
            trace = {
                "type": "surface", "x": section["x"], "y": section["y"], "z": section["z"],
                "colorscale": warm_colorscale, "showscale": False,
            }
        else:
            trace = {
                "type": "scatter3d", "mode": "markers",
                "x": section["x"], "y": section["y"], "z": section["z"],
                "marker": {
                    "size": 4, "color": section["z"], "colorscale": warm_colorscale,
                    "opacity": 0.85, "showscale": False,
                },
            }
        config = {"displaylogo": False, "responsive": True}
        return (
            f"Plotly.newPlot({json.dumps(plot_id)}, [{json.dumps(trace)}], "
            f"{json.dumps(common_layout)}, {json.dumps(config)});"
        )

    @staticmethod
    def _css() -> str:
        return """
:root {
  --bg: #F5F4EE;
  --card-bg: #FFFFFF;
  --border: #E8E4D9;
  --text: #262624;
  --text-muted: #83807A;
  --accent: #CC785C;
  --accent-dark: #B35F45;
  --accent-soft: #F3E4DC;
  --shadow: 0 1px 2px rgba(61,57,41,0.05), 0 4px 16px rgba(61,57,41,0.07);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.page { max-width: 1360px; margin: 0 auto; padding: 40px 32px 64px; }
.page-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 32px; }
.page-header h1 { font-size: 28px; font-weight: 700; margin: 0 0 6px; letter-spacing: -0.02em; }
.subtitle { margin: 0; color: var(--text-muted); font-size: 14px; }
.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
  gap: 24px;
}
.chart-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  padding: 22px 24px 18px;
  display: flex;
  flex-direction: column;
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.chart-card:hover { box-shadow: 0 2px 4px rgba(61,57,41,0.06), 0 12px 28px rgba(61,57,41,0.10); transform: translateY(-1px); }
.card-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
.card-header h2 { font-size: 15.5px; font-weight: 650; margin: 0 0 2px; letter-spacing: -0.01em; }
.card-source { margin: 0; font-size: 12px; color: var(--text-muted); }
.card-meta { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.chip {
  background: var(--accent-soft); color: var(--accent-dark); font-size: 11px; font-weight: 650;
  padding: 4px 9px; border-radius: 999px; letter-spacing: 0.02em; white-space: nowrap;
}
.card-stat { font-size: 12px; color: var(--text-muted); white-space: nowrap; }
.chart-body { position: relative; height: 340px; }
.plot-el { width: 100%; height: 100%; }
.page-footer { margin-top: 40px; text-align: center; font-size: 12.5px; color: var(--text-muted); }
@media (max-width: 520px) {
  .page { padding: 24px 16px 48px; }
  .dashboard-grid { grid-template-columns: 1fr; }
}
"""
