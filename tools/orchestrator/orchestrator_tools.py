from datetime import datetime, timezone
from typing import Optional

from rapidfuzz import fuzz

from agents.document import DocumentAgent
from agents.tabular import TabularAgent
from tools.hypothesis.hypothesis_tools import HypothesisTools
from tools.hypothesis.models import HypothesisResult
from tools.orchestrator.memory import LongTermMemory
from tools.orchestrator.models import FileRef, InvestigationEvent
from tools.reporting.models import ChartSpec
from tools.reporting.reporting_tools import ReportingTools
from tools.tabular.models import FileRef as TabularFileRef
from vectordb.chroma_store import ChromaVectorStore
from vectordb.reranker import CrossEncoderReranker


class OrchestratorTools:
    def __init__(self, catalog, state, vector_store=None, reranker=None, memory=None, storage=None):
        self.catalog = catalog
        self.state = state
        self.storage = storage
        self.workspace_id = "default"
        self._vector_store = vector_store
        self._reranker = reranker
        self.memory = memory or LongTermMemory()
        self.hypothesis_tools = HypothesisTools()
        self.reporting = ReportingTools(storage) if storage else None

    def list_files(self, workspace_id: str, filters: Optional[dict] = None, max_results: int = 20) -> list:
        """List files in the workspace matching a structured filter: name_contains, file_type
        (list, e.g. ["csv","pdf","table"]), uploaded_after/uploaded_before (ISO dates - call
        get_current_date first to resolve relative phrases like "4 months ago"), min_rows,
        max_rows, tags. Combine filters in one call rather than making several narrow calls."""
        filters = filters or {}
        results = [e for e in self.catalog.all() if self._matches(e, filters)]
        return results[:max_results]

    def _matches(self, entry, filters: dict) -> bool:
        if "name_contains" in filters and filters["name_contains"].lower() not in entry.filename.lower():
            return False
        if "file_type" in filters and entry.file_type not in filters["file_type"]:
            return False
        if "uploaded_after" in filters and entry.uploaded_at < datetime.fromisoformat(filters["uploaded_after"]):
            return False
        if "uploaded_before" in filters and entry.uploaded_at > datetime.fromisoformat(filters["uploaded_before"]):
            return False
        if "min_rows" in filters and (entry.row_count or 0) < filters["min_rows"]:
            return False
        if "max_rows" in filters and (entry.row_count or 0) > filters["max_rows"]:
            return False
        if "tags" in filters:
            entry_tags = entry.tags or []
            if not any(tag in entry_tags for tag in filters["tags"]):
                return False
        return True

    def search_files(self, workspace_id: str, query: str, max_results: int = 10) -> list:
        """Fuzzy search over filenames, for when the user's phrasing doesn't literally match a
        filename (e.g. "the churn numbers" vs "customer_retention_q3.csv"). Use this before
        list_files when you don't already know the exact file_id or filename."""
        scored = [(fuzz.partial_ratio(query.lower(), e.filename.lower()), e) for e in self.catalog.all()]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    def get_file_details(self, file_id: str):
        """Fetch full catalog metadata for one already-known file_id, without re-running a full
        list query."""
        entry = self.catalog.entries.get(file_id)
        if entry is None:
            raise ValueError(f"file_id '{file_id}' not found")
        return entry

    def list_tables(self, workspace_id: str, max_results: int = 20) -> list:
        """List every table in the workspace catalog - tables extracted from PDFs across all
        files, each with a file_id you can pass to invoke_tabular_agent. Shorthand for
        list_files(filters={"file_type": ["table"]})."""
        return self.list_files(workspace_id, filters={"file_type": ["table"]}, max_results=max_results)

    def list_file_formats(self, workspace_id: str) -> list:
        """List the distinct file types present in the workspace (e.g. ["csv", "pdf",
        "table"]). Use this to see what kinds of data exist before deciding how to filter
        list_files, especially early in an investigation."""
        return sorted({e.file_type for e in self.catalog.all()})

    def get_current_date(self) -> dict:
        """Get today's real date (UTC) and weekday name. Always call this before resolving any
        relative date the user mentions ("4 months ago", "last quarter", "since June") into the
        ISO dates list_files needs for uploaded_after/uploaded_before - never guess today's
        date or compute it from memory."""
        now = datetime.now(timezone.utc)
        return {"today": now.date().isoformat(), "weekday": now.strftime("%A")}

    def store_user_info(self, info: str) -> None:
        """Save a fact about the user or their preferences that should persist beyond this
        conversation (e.g. "user prefers quarterly breakdowns", "user's fiscal year starts in
        July"). Use this whenever the user states something worth remembering long-term - not
        for facts specific to only the current investigation, those belong in your own answer."""
        self.memory.remember(info)

    def recall_user_info(self) -> list:
        """Retrieve every fact previously saved with store_user_info, from any past session.
        Call this early if the user's request might be affected by something they told you
        before."""
        return self.memory.recall_all()

    async def invoke_tabular_agent(
        self,
        objective: str,
        assigned_files: list[FileRef],
        constraints: Optional[dict] = None,
        must_export: bool = False,
    ):
        """Delegate a data-analysis question to the Tabular Agent, scoped only to the given
        assigned_files (each {file_id, output_ref}). It runs its own DuckDB tool-calling loop
        in an isolated context and returns one compact TabularFindings - you never see its raw
        queries or intermediate results. Use for CSV/table data: aggregates, filters, joins,
        computed answers - including tables surfaced by the Document Agent via table_ref.

        Set must_export=True whenever the result needs to persist afterward (the user asked for
        a CSV, dashboard, or report) - this is enforced independently of how you word `objective`,
        so it survives even if you have to reword the objective on a retry after a failure. When
        True, this call raises instead of returning a fabricated or missing output_ref, so you
        know to retry rather than silently passing a fake reference on to generate_csv/
        generate_dashboard."""
        constraints = constraints or {}
        tabular_files = [self._to_tabular_file_ref(f) for f in assigned_files]
        agent = TabularAgent(tabular_files, storage=self.storage, workspace_id=self.workspace_id)

        effective_objective = objective
        if must_export:
            effective_objective += (
                "\n\nThis result MUST be persisted: your final computation must call "
                "query_data with persist=True (and a short name), and you must report its "
                "real output_ref string in your findings' artifact_refs."
            )

        result = await agent.run(effective_objective, constraints)

        if must_export:
            valid_refs = [
                ref for ref in result.artifact_refs
                if isinstance(ref, str) and (".parquet" in ref or "/" in ref or "\\" in ref)
            ]
            if not valid_refs:
                raise RuntimeError(
                    "invoke_tabular_agent was called with must_export=True but the Tabular "
                    "Agent did not return a real output_ref (it likely called query_data with "
                    "persist=False, or fabricated a placeholder artifact_ref). Retry with an "
                    "objective that explicitly tells it to call query_data with persist=True."
                )
            result.artifact_refs = valid_refs

        self._record_event("tabular", objective, result)
        return result

    async def invoke_document_agent(self, objective: str, assigned_files: list[FileRef], constraints: Optional[dict] = None):
        """Delegate a document-investigation question to the Document Agent, scoped only to
        the given assigned_files. It runs its own RAG tool-calling loop (search, verify, broad
        scans, table discovery) in an isolated context and returns one compact
        DocumentFindings - you never see its raw chunks. Use for PDF/text content: summaries,
        facts, quotes, or finding which tables exist in a document."""
        constraints = constraints or {}
        agent = DocumentAgent(assigned_files, vector_store=self._get_vector_store(), reranker=self._get_reranker())
        result = await agent.run(objective, constraints)
        self._record_event("document", objective, result)
        return result

    def generate_hypotheses(self, objective: str, context: dict, max_hypotheses: int = 5) -> HypothesisResult:
        """For complex or "why"-style objectives, generate and prioritize candidate
        explanations BEFORE delegating, so investigation effort targets the most likely
        directions first instead of exploring blindly. context = {"available_files": [... from
        a prior list_files call ...], "known_findings": [... optional, only if this is a
        follow-up round ...]}. Skip this for simple, direct questions - go straight to
        invoke_tabular_agent/invoke_document_agent instead."""
        result = self.hypothesis_tools.generate_hypotheses(objective, context, max_hypotheses)
        self._record_event("hypothesis", objective, result)
        self.state.open_questions = [h.statement for h in result.hypotheses]
        return result

    def generate_csv(self, output_ref: str, name: Optional[str] = None) -> str:
        """Convert an existing data artifact (an output_ref you got from a table_ref or from
        a tabular agent's persisted-query artifact_ref) into a CSV file. Use this when the user
        asks for a CSV/spreadsheet, not a written report or a dashboard. Creates a new folder
        under today's date named after `name` (a short label for this request, e.g.
        "q3_revenue_by_region") and writes the CSV there together with a copy of the source
        data file, so the whole request's output lives in one place. Returns the CSV file path
        - report it in your final answer and in artifact_refs."""
        if self.reporting is None:
            raise RuntimeError("no storage configured, cannot generate files")
        return self.reporting.generate_csv(output_ref, name)

    def generate_markdown_report(
        self,
        title: str,
        objective: str,
        summary: str,
        findings: list,
        open_questions: Optional[list] = None,
        name: Optional[str] = None,
    ) -> str:
        """Build a markdown report file from your OWN synthesized investigation results - pass
        your own summary and findings text (short strings you write), not raw tool output. Use
        this when the user asks for a written report/document, not a CSV or dashboard. Creates a
        new folder under today's date named after `name` (falls back to a slug of title) and
        writes the report there. Returns the file path - report it in your final answer and in
        artifact_refs."""
        if self.reporting is None:
            raise RuntimeError("no storage configured, cannot generate files")
        return self.reporting.generate_markdown_report(title, objective, summary, findings, open_questions, name)

    def generate_dashboard(self, title: str, sections: list[ChartSpec], name: Optional[str] = None) -> str:
        """Build a single-file HTML dashboard with charts from one or more existing data
        artifacts (output_refs from table_refs or a persisted query_data call). Use this when the
        user asks for a dashboard or visualization, not a CSV or written report.

        Each item in `sections` is a ChartSpec: {output_ref, chart_type, ...column names...}.
        You never pass or see actual data values here - only an output_ref (a file path) and
        column names you already know from a Tabular Agent's findings; the real numbers are
        read straight from the parquet file when the dashboard is built.

        chart_type options and which column names each needs:
        - "bar" / "line": EITHER label_column + value_columns (1+ numeric series - omit both to
          auto-pick the first non-numeric column and up to 5 numeric columns) OR, when the
          result has TWO grouping columns and one metric (e.g. Age, Gender, Customer Count),
          label_column + series_column + value_column - this produces one bar/line per distinct
          series_column value grouped along label_column (e.g. label_column="Age",
          series_column="Gender", value_column="Customer Count"). Use this whenever a Tabular
          Agent's result has more than one grouping column - never pick just one and drop the
          other.
        - "timeline": time_column (required) plus EITHER value_columns (wide data - one series
          per column) OR series_column + value_column (long/tidy data - one series per distinct
          value in series_column, e.g. columns date, job_title, count ->
          series_column="job_title", value_column="count").
        - "scatter3d" / "surface": x_column, y_column, z_column (all required). "surface" needs
          every (x, y) combination present in the data to build a valid grid - use "scatter3d"
          instead if that can't be guaranteed.

        Creates a new folder under today's date named after `name` (falls back to a slug of
        title) and writes the dashboard there together with copies of every source data file
        that fed it. Returns the file path - report it in your final answer and in
        artifact_refs."""
        if self.reporting is None:
            raise RuntimeError("no storage configured, cannot generate files")
        return self.reporting.generate_dashboard(title, sections, name)

    def _record_event(self, event_type: str, objective: str, result) -> None:
        event = InvestigationEvent(
            event_type=event_type,
            objective=objective,
            result=result,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.state.add_event(event)

    def _to_tabular_file_ref(self, file_ref) -> TabularFileRef:
        entry = self.catalog.entries.get(file_ref.file_id)
        filename = entry.filename if entry else ""
        return TabularFileRef(file_id=file_ref.file_id, output_ref=file_ref.output_ref, filename=filename)

    def _get_vector_store(self):
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore()
        return self._vector_store

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker
