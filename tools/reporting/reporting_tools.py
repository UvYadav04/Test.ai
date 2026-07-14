import json
import os
import re
import shutil
import uuid
from datetime import date
from typing import Optional

import pandas as pd


class ReportingTools:
    def __init__(self, storage, output_dir: str = "data/reports"):
        self.storage = storage
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_csv(self, output_ref: str, name: Optional[str] = None) -> str:
        """Convert an existing data artifact (an output_ref from a table_ref or from
        export_query) into a CSV file. Creates a new dated folder (today's date + name) and
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

    def generate_dashboard(self, title: str, output_refs: list, name: Optional[str] = None) -> str:
        """Build a single self-contained HTML dashboard (charts rendered client-side via
        Chart.js) from one or more existing data artifacts (output_refs). Use this when the
        user wants a visual dashboard, not a CSV or written report. Creates a new dated folder
        (today's date + name, falling back to a slug of title) and writes the dashboard there,
        alongside copies of every source data file that fed it."""
        folder = self._new_folder(name or title, "dashboard")
        sections = []
        for ref in output_refs:
            sections.append(self._chart_section(self.storage.read(ref), ref))
            self._copy_source(ref, folder)

        html = self._render_html(title, sections)
        path = os.path.join(folder, "dashboard.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

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
    def _chart_section(dataframe: pd.DataFrame, output_ref: str) -> dict:
        numeric_cols = [c for c in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[c])]
        label_col = next((c for c in dataframe.columns if c not in numeric_cols), dataframe.columns[0])
        rows = dataframe.head(50)
        return {
            "title": output_ref,
            "labels": rows[label_col].astype(str).tolist(),
            "datasets": [{"label": col, "data": rows[col].tolist()} for col in numeric_cols[:5]],
        }

    @staticmethod
    def _render_html(title: str, sections: list) -> str:
        canvases = "\n".join(f'<canvas id="chart{i}"></canvas>' for i in range(len(sections)))
        scripts = "\n".join(
            f"""
            new Chart(document.getElementById('chart{i}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(section["labels"])},
                    datasets: {json.dumps(section["datasets"])}
                }},
                options: {{ responsive: true, plugins: {{ title: {{ display: true, text: {json.dumps(section["title"])} }} }} }}
            }});
            """
            for i, section in enumerate(sections)
        )
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; }}
canvas {{ margin-bottom: 50px; }}
</style>
</head>
<body>
<h1>{title}</h1>
{canvases}
<script>
{scripts}
</script>
</body>
</html>
"""
