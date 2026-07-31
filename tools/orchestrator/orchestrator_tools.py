from datetime import datetime, timezone
from typing import Optional

from rapidfuzz import fuzz

from agents.document import DocumentAgent
from agents.tabular import TabularAgent
from sandbox.path_resolver import InvalidArtifactIdError, get_parquet_path, validate_segment
from tools.hypothesis.hypothesis_tools import HypothesisTools
from tools.hypothesis.models import HypothesisResult
from tools.document.document_processor import DocumentProcessor
from tools.document.metadata import build_document_metadata_brief
from tools.orchestrator.file_catalog import is_tabular_output_ref
from tools.orchestrator.memory import LongTermMemory
from tools.orchestrator.models import FileRef, FinalResultCollector, InvestigationEvent
from tools.reporting.models import ChartSpec
from tools.reporting.reporting_tools import ReportingTools
from tools.tabular.models import FileRef as TabularFileRef
from vectordb.chroma_store import ChromaVectorStore
from vectordb.reranker import DeepInfraReranker


def _looks_like_file_id(ref: str) -> bool:
    try:
        validate_segment(ref, "artifact_ref")
        return True
    except InvalidArtifactIdError:
        return False


class OrchestratorTools:
    def __init__(
        self, catalog, state, vector_store=None, reranker=None, memory=None, storage=None,
        reports_dir: str = "data/reports", chat_id: str = "default", sandbox_manager=None,
        result_collector: FinalResultCollector = None, chart_capacity_checker=None,
    ):
        self.catalog = catalog
        self.state = state
        self.result_collector = result_collector or FinalResultCollector()
        self.storage = storage
        self.workspace_id = "default"
        self.chat_id = chat_id
        self.sandbox_manager = sandbox_manager
        self.chart_capacity_checker = chart_capacity_checker
        self._vector_store = vector_store
        self._reranker = reranker
        self.memory = memory or LongTermMemory()
        self.hypothesis_tools = HypothesisTools()
        self.reporting = ReportingTools(storage, output_dir=reports_dir) if storage else None
        self.reports_dir = reports_dir
        self.on_event = None
        self._last_transform_script: Optional[str] = None
        self._last_tabular_file_ids: list = []

    def list_files(self, workspace_id: str, filters: Optional[dict] = None, max_results: int = 20) -> list:
        """
        List queryable workspace files.

        Supported filters: name_contains, file_type, uploaded_after, uploaded_before,
        min_rows, max_rows, tags.

        Combine filters into one call. Resolve relative dates with get_current_date.
        Excel workbooks appear as sheet tables (file_type="table"). PDF tables are
        available only through the document agent.
        """
        filters = filters or {}
        results = [e for e in self.catalog.browsable() if self._matches(e, filters)]
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
        list_files when you don't already know the exact file_id or filename. Same visibility
        rules as list_files - see its docstring for what's excluded and why."""
        scored = [
            (fuzz.partial_ratio(query.lower(), e.filename.lower()), e) for e in self.catalog.browsable()
        ]
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
        """
        List queryable Excel sheet tables.

        Returns each worksheet with its own file_id for invoke_tabular_agent.
        Use this instead of a workbook's file_id when analyzing Excel files.
        PDF tables are excluded; access them through the document agent.
        """
        return self.list_files(workspace_id, filters={"file_type": ["table"]}, max_results=max_results)

    def list_file_formats(self, workspace_id: str) -> list:
        """List the distinct queryable file types in the workspace (e.g. csv, pdf,
        table). Use this before list_files when you need to discover available
        data types. "table" refers only to Excel sheet tables, not PDF tables."""
        return sorted({e.file_type for e in self.catalog.browsable()})

    def get_current_date(self) -> dict:
        """
        Get the current UTC date.
        Call this before resolving any relative date into an absolute date. Never
        assume today's date from memory.
        """
        now = datetime.now(timezone.utc)
        return {"today": now.date().isoformat(), "weekday": now.strftime("%A")}

    def recall_user_info(self) -> list:
        """Retrieve every long-term user fact/preference saved from any past session (see
        worker_service.tasks.investigation.update_chat_memory, which extracts these
        automatically after each completed turn - there is no explicit "save" tool anymore).
        This is already precomputed into your task prompt at the start of every investigation
        (see OrchestratorAgent._context_brief) - call this again only if you need to double-check
        it mid-run."""
        return self.memory.recall_all()

    async def invoke_tabular_agent(
        self,
        objective: str,
        assigned_files: list[FileRef],
        constraints: Optional[dict] = None,
        must_export: bool = False,
    ):
        """Delegate structured data analysis to the Tabular Agent.

            Use for CSV or table data requiring filtering, aggregation, joins, calculations,
            or analysis. Pass only valid file_ids in assigned_files. The agent returns
            compact findings, not raw data or code.

            Set must_export=True whenever the user needs reusable outputs (e.g. CSV,
            dashboard, or report).

            If the objective calls for a chart, the Tabular Agent generates and saves it itself
            (via its own create_visualizations tool) as part of this same call - you don't
            generate, validate, or lay out charts yourself. The returned findings' `charts` field
            already lists every chart it made (chart_id, artifact_file_id, chart_type, title,
            location) - just mention them in your final answer; no further tool call is needed to
            produce or attach them."""
        constraints = constraints or {}
        self.state.selected_files.extend(f.file_id for f in assigned_files)
        tabular_files = [self._to_tabular_file_ref(f) for f in assigned_files]
        agent = TabularAgent(
            tabular_files, storage=self.storage, workspace_id=self.workspace_id,
            chat_id=self.chat_id, sandbox_manager=self.sandbox_manager,
            reports_dir=self.reports_dir, chart_capacity_checker=self.chart_capacity_checker,
        )

        effective_objective = objective
        if must_export:
            effective_objective += (
                "\n\nThis result MUST be persisted: your final computation must call "
                "save(df, name) inside run_python, and you must report its real file_id "
                "in your findings' artifact_refs."
            )

        result = await agent.run(effective_objective, constraints, on_event=self.on_event)

        if agent.last_transform_script:
            self._last_transform_script = agent.last_transform_script
            self._last_tabular_file_ids = agent.last_transform_file_ids

        if must_export:
            valid_refs = [ref for ref in result.artifact_refs if isinstance(ref, str) and _looks_like_file_id(ref)]
            if not valid_refs:
                raise RuntimeError(
                    "invoke_tabular_agent was called with must_export=True but the Tabular "
                    "Agent did not return a real file_id (it likely never called save(), or "
                    "fabricated a placeholder artifact_ref). Retry with an objective that "
                    "explicitly tells it to call save(df, name) inside run_python."
                )
            result.artifact_refs = valid_refs

        self.result_collector.add_tabular_findings(
            result, "invoke_tabular_agent", [f.file_id for f in assigned_files],
        )
        self._record_event("tabular", objective, result)
        return result

    async def invoke_document_agent(self, objective: str, assigned_files: list[FileRef], constraints: Optional[dict] = None):
        """Delegate a TARGETED document question to the Document Agent, scoped only to the
        given assigned_files. It runs its own RAG tool-calling loop (search, verify, table
        discovery) in an isolated context and returns one compact DocumentFindings - you never
        see its raw chunks. Use for a specific fact, quote, section-specific question, or
        finding which tables exist in a document.

        Do NOT use this for whole-document tasks (summarize, explain, executive summary, key
        takeaways, find anomalies/risks, extract action items, FAQ, insights) - use
        invoke_document_processor for those instead."""
        constraints = constraints or {}
        self.state.selected_files.extend(f.file_id for f in assigned_files)
        vector_store = self._get_vector_store()
        metadata_brief = build_document_metadata_brief(
            self.catalog, vector_store, [f.file_id for f in assigned_files],
        )
        agent = DocumentAgent(assigned_files, vector_store=vector_store, reranker=self._get_reranker())
        result = await agent.run(objective, constraints, on_event=self.on_event, metadata_brief=metadata_brief)
        self.result_collector.add_document_findings(
            result, "invoke_document_agent", [f.file_id for f in assigned_files],
        )
        self._record_event("document", objective, result)
        return result

    async def invoke_document_processor(
        self,
        objective: str,
        assigned_files: list[FileRef],
        constraints: Optional[dict] = None,
    ):
        """Deterministic whole-document analysis.

        Use this instead of `invoke_document_agent` whenever the user's request
        requires understanding the entire assigned document(s), not retrieving
        specific information.

        Examples:
        - Summarize or explain a document
        - Generate an executive summary or key takeaways
        - Find anomalies, risks, or insights
        - Extract action items
        - Any other whole-document analysis

        Do NOT use this for targeted questions, keyword lookups, section-specific
        queries, or iterative investigations. Use `invoke_document_agent` instead.

        `objective` is a free-form instruction written by the orchestrator. It is
        passed directly to the analysis model, so make it clear, specific, and
        complete.

        The processor automatically receives metadata for assigned documents,
        retrieves all chunks in order, processes them in token-limited batches,
        merges the intermediate results, and performs one final LLM call to produce
        the final response.
        """
        constraints = constraints or {}
        self.state.selected_files.extend(f.file_id for f in assigned_files)
        processor = DocumentProcessor(assigned_files, vector_store=self._get_vector_store())
        result = await processor.run(objective, constraints, on_event=self.on_event)
        self.result_collector.add_document_findings(
            result, "invoke_document_processor", [f.file_id for f in assigned_files],
        )
        self._record_event("document_processor", objective, result)
        return result

    def generate_hypotheses(self, objective: str, context: dict, max_hypotheses: int = 5) -> HypothesisResult:
        """Generate and prioritize hypotheses for complex investigations.

            Use for open-ended or "why" questions before delegating to other agents.
            Provide available_files (file_id, filename, file_type) and optional
            known_findings for follow-up investigations. Skip this for direct
            questions; delegate straight to the appropriate agent."""
        result = self.hypothesis_tools.generate_hypotheses(objective, context, max_hypotheses)
        self._record_event("hypothesis", objective, result)
        self.state.open_questions = [h.statement for h in result.hypotheses]
        return result

    def generate_csv(self, file_id: str, name: Optional[str] = None) -> str:
        """Create a CSV from an existing data artifact.

        Use when the user requests a CSV or spreadsheet, not a report or dashboard.
        Provide the source file_id and a short output name. Returns the generated
        CSV file path."""
        if self.reporting is None:
            raise RuntimeError("no storage configured, cannot generate files")
        path = self.reporting.generate_csv(self.workspace_id, self._resolve_file_id(file_id), name)
        self.result_collector.add_artifact(path, kind="report", format="csv", source_tool="generate_csv")
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
        """Build a markdown report file from your OWN synthesized investigation results - pass
        your own summary and findings text (short strings you write), not raw tool output. Use
        this when the user asks for a written report/document, not a CSV or dashboard. Creates a
        new folder under today's date named after `name` (falls back to a slug of title) and
        writes the report there. Returns the file path - report it in your final answer and in
        artifact_refs."""
        if self.reporting is None:
            raise RuntimeError("no storage configured, cannot generate files")
        path = self.reporting.generate_markdown_report(title, objective, summary, findings, open_questions, name)
        self.result_collector.add_artifact(
            path, kind="report", format="markdown", source_tool="generate_markdown_report",
        )
        return path

    def generate_dashboard(self, title: str, sections: list[ChartSpec], name: Optional[str] = None) -> str:
        """Build a PERSISTENT, auto-refreshing dashboard - use this only when the user explicitly
        wants something that stays live/updated over time (e.g. "keep this dashboard updated",
        "a live view of X"), not for an ordinary chart in your answer. An ordinary chart request
        is already handled entirely by invoke_tabular_agent's own create_visualizations tool (see
        its docstring) with no action needed from you here - do not call generate_dashboard for
        that case.

        Requires a preceding invoke_tabular_agent call THIS investigation (it must have run
        run_python) - the dashboard is built from that call's saved data and stays refreshable
        against it later with no LLM involvement (see shared/models/dashboard.py).

        Each section is a ChartSpec containing a file_id, chart_type, and the
        required column names:
        - bar/line: label_column + value_column(s), or label_column + series_column + value_column
        - timeline: time_column + value_column(s), or time_column + series_column + value_column
        - scatter3d/surface: x_column, y_column, z_column

        Returns the generated dashboard's manifest path."""
        if self.reporting is None:
            raise RuntimeError("no storage configured, cannot generate files")

        if not self._last_transform_script:
            raise RuntimeError(
                "generate_dashboard requires a preceding invoke_tabular_agent call THIS "
                "investigation that ran run_python, so the dashboard can stay refreshable "
                "against the same script/file_ids later. Call invoke_tabular_agent first, then "
                "generate_dashboard right after."
            )

        resolved_sections = [self._resolve_chart_spec(section) for section in sections]
        manifest_path = self.reporting.generate_realtime_dashboard_bundle(
            self.workspace_id, title, resolved_sections, self._last_transform_script,
            self._last_tabular_file_ids, name,
        )
        self.result_collector.add_artifact(
            manifest_path, kind="dashboard_bundle", source_tool="generate_dashboard",
        )
        return manifest_path

    def _known_artifact_refs(self) -> list:
        refs = []
        for event in self.state.completed_tasks:
            for ref in getattr(event.result, "artifact_refs", None) or []:
                if isinstance(ref, str) and ref not in refs:
                    refs.append(ref)
        return refs

    def _artifact_path(self, file_id: str) -> Optional[str]:
        if self.storage is None:
            return None
        try:
            return get_parquet_path(self.storage.root_dir, self.workspace_id, file_id)
        except (InvalidArtifactIdError, AttributeError, TypeError):
            return None

    def _resolve_file_id(self, file_id: str) -> str:
        path = self._artifact_path(file_id)
        if path is not None and self.storage.exists(path):
            return file_id

        known = self._known_artifact_refs()
        if not known:
            return file_id

        best_ref, best_score = None, 0.0
        for ref in known:
            score = fuzz.ratio(file_id, ref)
            if score > best_score:
                best_ref, best_score = ref, score

        if best_ref is not None and best_score >= 90:
            return best_ref

        raise ValueError(
            f"file_id '{file_id}' does not exist and doesn't closely match any artifact this "
            f"investigation has actually produced. Real file_ids so far: {known}. Copy one of "
            "these exactly rather than retyping it."
        )

    def _resolve_chart_spec(self, raw) -> ChartSpec:
        spec = ReportingTools._to_chart_spec(raw)
        spec.file_id = self._resolve_file_id(spec.file_id)
        return spec

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
        if entry is None:
            raise ValueError(f"file_id '{file_ref.file_id}' not found in catalog")
        if not is_tabular_output_ref(entry.output_ref):
            raise ValueError(
                f"file_id '{file_ref.file_id}' ('{entry.filename}') has no queryable tabular "
                "data of its own - it's a PDF or xlsx workbook's main entry, not an actual "
                "table. Call list_tables to find the individual table file_id(s) extracted from "
                "it, and pass those to invoke_tabular_agent instead."
            )
        return TabularFileRef(file_id=entry.file_id, filename=entry.filename)

    def _get_vector_store(self):
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore()
        return self._vector_store

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = DeepInfraReranker()
        return self._reranker
