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
    # {file_id: {row_count, columns, dtypes, column_kinds, preview}} for every artifact in
    # artifact_refs - captured straight from the sandbox's own save() response (see
    # tools/tabular/tabular_tools.py's run_python and sandbox/execution_engine.py's save()),
    # never re-derived from the LLM's own transcription. Lets the orchestrator (and a later
    # generate_dashboard/generate_csv call) see real schema and numeric/datetime/categorical
    # column_kinds hints alongside the file_id, instead of only a bare id it would otherwise
    # have to re-inspect via another tool round trip.
    artifact_metadata: dict = field(default_factory=dict)
    # Analysis SEMANTICS, not just schema: which column is the category/axis, which is the
    # metric, what chart type fits, what to title it - things only the Tabular Agent actually
    # knows (it read the objective, planned the query, and interpreted the result; the
    # orchestrator never sees any of that). Populated only via TabularTools.propose_visualization
    # (see there) - a real, validated tool call the model makes, never inferred from summary text
    # or column names after the fact. Each entry is a plain dict using EXACTLY
    # tools.reporting.models.ChartSpec's field names (file_id, chart_type, title, label_column,
    # value_columns, time_column, series_column, value_column, x_column, y_column, z_column) -
    # deliberately, so the orchestrator can pass entries from this list straight through as
    # generate_dashboard's `sections` argument with zero translation/reinterpretation. Empty when
    # the objective didn't call for a chart, or the model judged nothing here needs one.
    visualization_plan: list = field(default_factory=list)


@dataclass
class DocumentFindings:
    summary: str
    artifact_refs: list = field(default_factory=list)
    source_refs: list = field(default_factory=list)


@dataclass
class InvestigationEvent:
    event_type: str  # "tabular" | "document" | "hypothesis"
    objective: str
    result: object
    timestamp: str


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
    artifact_refs: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    files_used: list = field(default_factory=list)
