from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class FileCatalogEntry:
    file_id: str
    filename: str
    file_type: str
    uploaded_at: datetime
    size_bytes: int
    output_ref: str = ""
    row_count: Optional[int] = None
    page_count: Optional[int] = None
    columns: Optional[list] = None
    tags: Optional[list] = None


@dataclass
class FileRef:
    file_id: str


@dataclass
class TabularFindings:
    summary: str
    artifact_refs: list = field(default_factory=list)
    artifact_metadata: dict = field(default_factory=dict)
    charts: list = field(default_factory=list)
    follow_up_questions: list = field(default_factory=list)


@dataclass
class DocumentFindings:
    summary: str
    artifact_refs: list = field(default_factory=list)
    source_refs: list = field(default_factory=list)
    follow_up_questions: list = field(default_factory=list)


@dataclass
class InvestigationEvent:
    event_type: str
    objective: str
    result: object
    timestamp: str


@dataclass
class FinalResultCollector:
    chart_paths: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    files_used: list = field(default_factory=list)

    def add_chart(self, location: str, **meta) -> None:
        if not location or any(c["location"] == location for c in self.chart_paths):
            return
        self.chart_paths.append({"type": "chart", "location": location, **meta})

    def add_artifact(self, ref: str, kind: str, **meta) -> None:
        if not ref or any(a["ref"] == ref for a in self.artifacts):
            return
        self.artifacts.append({"type": kind, "ref": ref, **meta})

    def add_files_used(self, file_ids: list) -> None:
        for file_id in file_ids or []:
            if file_id and file_id not in self.files_used:
                self.files_used.append(file_id)

    def add_tabular_findings(self, findings, source_tool: str, assigned_file_ids: list) -> None:
        self.add_files_used(assigned_file_ids)
        chart_locations = set()
        for chart in findings.charts:
            chart_locations.add(chart["location"])
            self.add_chart(
                chart["location"], chart_id=chart["chart_id"], chart_type=chart["chart_type"],
                title=chart["title"], artifact_file_id=chart.get("artifact_file_id"),
                source_tool=source_tool,
            )
        for ref in findings.artifact_refs:
            if ref in chart_locations:
                continue
            self.add_artifact(ref, kind="table", source_tool=source_tool)

    def add_document_findings(self, findings, source_tool: str, assigned_file_ids: list) -> None:
        self.add_files_used(assigned_file_ids)
        for ref in findings.artifact_refs:
            self.add_artifact(ref, kind="document_artifact", source_tool=source_tool)


@dataclass
class InvestigationState:
    session_id: str
    objective: str
    constraints: dict = field(default_factory=dict)
    selected_files: list = field(default_factory=list)
    active_tasks: list = field(default_factory=list)
    completed_tasks: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    status: str = "in_progress"

    def add_event(self, event: InvestigationEvent) -> None:
        self.completed_tasks.append(event)
        self.findings.append(event.result)

    def summary(self) -> str:
        lines = [f"Investigation for: {self.objective}", f"Status: {self.status}"]
        for event in self.completed_tasks:
            result_summary = getattr(event.result, "summary", str(event.result))
            lines.append(f"- [{event.event_type}] {event.objective} -> {result_summary}")
        if self.open_questions:
            lines.append("Open questions: " + "; ".join(self.open_questions))
        return "\n".join(lines)


@dataclass
class OrchestratorResult:
    final_answer: str
    chart_paths: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    files_used: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    follow_up_questions: list = field(default_factory=list)

    @property
    def artifact_refs(self) -> list:
        return [c["location"] for c in self.chart_paths] + [a["ref"] for a in self.artifacts]
