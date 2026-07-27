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
class FinalResultCollector:
    """Accumulates everything a sub-agent/tool call has actually produced during ONE
    investigation, updated immediately as each call's result comes back - not re-derived at the
    end from LLM transcript text, and not lost just because a LATER tool call in the same run
    raises. Created once per investigation (worker_service.tasks.investigation.run_investigation)
    and threaded down through OrchestratorAgent -> OrchestratorTools, and separately into the
    direct-route helpers (_run_tabular_direct/_run_document_direct) so every producing call -
    invoke_tabular_agent, invoke_document_agent, invoke_document_processor,
    generate_csv/generate_markdown_report/generate_dashboard, or a direct-routed
    TabularAgent/DocumentAgent - registers here right when its own result comes back, regardless
    of what the orchestrator does afterward or whether the run eventually crashes.

    Kept as three separate lists (not one flat artifact_refs) so a consumer never has to re-guess
    "is this a chart or a report" from a file extension the way worker_service used to:
    - chart_paths: local HTML paths for charts already rendered (TabularTools.
      create_visualizations output). Each entry: {type: "chart", location, chart_id, chart_type,
      title, artifact_file_id, source_tool}.
    - artifacts: everything else this investigation actually produced - reports (csv/markdown),
      a real-time dashboard's manifest.json, a saved tabular data artifact, or a table_ref
      surfaced by a Document agent. Each entry: {type: "report"|"dashboard_bundle"|"table"|
      "document_artifact", ref, source_tool, ...extra metadata}.
    - files_used: file_ids this investigation actually read from, for the "these files were
      used" chips the client shows under an assistant message.

    Deliberately dumb (dict bags, not typed dataclasses per entry) - the shape each tool needs to
    attach varies (a chart has chart_type/title, a CSV export doesn't), and nothing downstream
    needs more than dict access.
    """

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
        """Shared by OrchestratorTools.invoke_tabular_agent and worker_service's
        _run_tabular_direct (the direct-route path bypasses the Orchestrator/OrchestratorTools
        entirely) so both register a TabularFindings the exact same way. findings.charts entries
        are already confirmed-rendered (see TabularTools.create_visualizations) - never trust an
        LLM transcription of them. findings.artifact_refs also includes each chart's own
        location (see TabularAgent.run()'s note) - skipped here since add_chart already covers
        it, so a chart is never double-registered as a "table" artifact too."""
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
        """Shared by OrchestratorTools.invoke_document_agent/invoke_document_processor and
        worker_service's _run_document_direct - same reasoning as add_tabular_findings above."""
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
    # Typed, straight off a FinalResultCollector (see above) - chart_paths/artifacts are kept
    # separate rather than flattened, so worker_service's persistence step never has to re-guess
    # "chart vs report vs dashboard" from a file extension.
    chart_paths: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    files_used: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)

    @property
    def artifact_refs(self) -> list:
        """Back-compat flat view (every chart location + every artifact ref) for any caller not
        yet updated to read chart_paths/artifacts directly - e.g. worker_service's
        run_investigation, which only ever needed "what did this investigation produce" as a
        flat list to pass through as files_created when it enqueues the update_chat_memory job."""
        return [c["location"] for c in self.chart_paths] + [a["ref"] for a in self.artifacts]
