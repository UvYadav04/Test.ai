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
    # Metadata for charts that have ALREADY been generated and saved - not a plan for the
    # orchestrator to act on. Populated only via TabularTools.create_visualizations (see there),
    # which validates, renders, and saves each chart itself; each entry's "location" is also
    # folded into artifact_refs above so worker_service's existing artifact-persistence pipeline
    # (see worker_service/tasks/investigation.py's _persist_artifacts) uploads it and creates a
    # Chart doc with no dashboard/chart-generation step of its own. The orchestrator only reads
    # this to know what to mention in its final answer (chart_id, artifact_file_id, chart_type,
    # title, location per chart) - it never generates, validates, or lays out charts itself.
    # Empty when the objective didn't call for a chart, or the model judged nothing here needs one.
    charts: list = field(default_factory=list)


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
