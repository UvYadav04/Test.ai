from datetime import datetime, timezone
from typing import Optional

from rapidfuzz import fuzz

from agents.document import DocumentAgent
from agents.tabular import TabularAgent
from sandbox.path_resolver import InvalidArtifactIdError, get_parquet_path, validate_segment
from tools.hypothesis.hypothesis_tools import HypothesisTools
from tools.hypothesis.models import HypothesisResult
from tools.orchestrator.file_catalog import is_tabular_output_ref
from tools.orchestrator.memory import LongTermMemory
from tools.orchestrator.models import FileRef, InvestigationEvent
from tools.reporting.models import ChartSpec
from tools.reporting.reporting_tools import ReportingTools
from tools.tabular.models import FileRef as TabularFileRef
from vectordb.chroma_store import ChromaVectorStore
from vectordb.reranker import CrossEncoderReranker


def _looks_like_file_id(ref: str) -> bool:
    try:
        validate_segment(ref, "artifact_ref")
        return True
    except InvalidArtifactIdError:
        return False


class OrchestratorTools:
    def __init__(
        self, catalog, state, vector_store=None, reranker=None, memory=None, storage=None,
        reports_dir: str = "data/reports", investigation_id: str = "default", sandbox_manager=None,
    ):
        self.catalog = catalog
        self.state = state
        self.storage = storage
        self.workspace_id = "default"
        # Handed straight through to invoke_tabular_agent -> TabularAgent -> TabularTools ->
        # PythonSandbox, so every Tabular Agent this orchestrator delegates to during THIS
        # investigation reuses the same persistent sandbox container - see
        # sandbox/sandbox_manager.py's docstring for why investigation_id (not workspace_id) is
        # the right cache key.
        self.investigation_id = investigation_id
        # The real SandboxManager instance - see OrchestratorAgent.__init__'s note on why this
        # is passed explicitly instead of re-resolved via get_manager() at this layer.
        self.sandbox_manager = sandbox_manager
        self._vector_store = vector_store
        self._reranker = reranker
        self.memory = memory or LongTermMemory()
        self.hypothesis_tools = HypothesisTools()
        self.reporting = ReportingTools(storage, output_dir=reports_dir) if storage else None
        # Set per-run by OrchestratorAgent.run() (there's no `run` call on
        # this class itself to hand it in through). Forwarded into the
        # delegated Tabular/Document agent's own run() so its tool calls
        # (run_python, search_documents, ...) stream as events too, not just
        # invoke_tabular_agent/invoke_document_agent's own start event.
        self.on_event = None
        # Most recent tabular run_python call's code/file_ids, captured off the
        # TabularAgent instance after each invoke_tabular_agent call - see the note
        # there. Only read by generate_dashboard(real_time=True).
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

    def store_user_info(self, info: str) -> None:
        """Save a long-term user fact or preference.

        Use for information that should persist across conversations, not
        temporary or task-specific details."""
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
        """Delegate structured data analysis to the Tabular Agent.

            Use for CSV or table data requiring filtering, aggregation, joins, calculations,
            or analysis. Pass only valid file_ids in assigned_files. The agent returns
            compact findings, not raw data or code.

            Set must_export=True whenever the user needs reusable outputs (e.g. CSV,
            dashboard, or report).

            The returned findings may include visualization_plan: pre-worked-out chart
            sections (file_id, chart_type, title, and the right column names) for artifacts the
            Tabular Agent judged should be visualized. If you're building a dashboard from this
            call's output, pass visualization_plan straight to generate_dashboard's `sections` -
            see that tool's own docstring."""
        constraints = constraints or {}
        self.state.selected_files.extend(f.file_id for f in assigned_files)
        tabular_files = [self._to_tabular_file_ref(f) for f in assigned_files]
        agent = TabularAgent(
            tabular_files, storage=self.storage, workspace_id=self.workspace_id,
            investigation_id=self.investigation_id, sandbox_manager=self.sandbox_manager,
        )

        effective_objective = objective
        if must_export:
            effective_objective += (
                "\n\nThis result MUST be persisted: your final computation must call "
                "save(df, name) inside run_python, and you must report its real file_id "
                "in your findings' artifact_refs."
            )

        result = await agent.run(effective_objective, constraints, on_event=self.on_event)

        # Stashed off the agent object itself (not part of `result`/TabularFindings - see
        # TabularAgent.__init__'s note) so generate_dashboard(real_time=True) can find the
        # script that produced this call's data without it ever entering the orchestrator
        # LLM's own context. Each invoke_tabular_agent call overwrites this - a real-time
        # dashboard's sections must all come from the SAME (most recent) tabular call, see
        # generate_dashboard's docstring.
        if agent.last_transform_script:
            self._last_transform_script = agent.last_transform_script
            self._last_tabular_file_ids = agent.last_transform_file_ids

        if must_export:
            # agent.py's _real_refs already confirmed each of these is a file_id that really
            # exists on disk (see agents/tabular/agent.py) - this is just the final "did it
            # look like an id at all" shape check, now against a short [0-9a-zA-Z_-]+ token
            # instead of the old ".parquet"/"/"/"\\" path-shape sniffing.
            valid_refs = [ref for ref in result.artifact_refs if isinstance(ref, str) and _looks_like_file_id(ref)]
            if not valid_refs:
                raise RuntimeError(
                    "invoke_tabular_agent was called with must_export=True but the Tabular "
                    "Agent did not return a real file_id (it likely never called save(), or "
                    "fabricated a placeholder artifact_ref). Retry with an objective that "
                    "explicitly tells it to call save(df, name) inside run_python."
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
        self.state.selected_files.extend(f.file_id for f in assigned_files)
        agent = DocumentAgent(assigned_files, vector_store=self._get_vector_store(), reranker=self._get_reranker())
        result = await agent.run(objective, constraints, on_event=self.on_event)
        self._record_event("document", objective, result)
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
        return self.reporting.generate_csv(self.workspace_id, self._resolve_file_id(file_id), name)

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

    def generate_dashboard(
        self, title: str, sections: list[ChartSpec], name: Optional[str] = None, real_time: bool = False,
    ) -> str:
        """Build a dashboard from one or more data artifacts.

            Use when the user requests a dashboard or visualizations, not a CSV or
            written report.

            If invoke_tabular_agent's result for the artifact(s) you're using included a
            visualization_plan, pass those entries here as `sections` VERBATIM - each one is
            already a complete, validated ChartSpec-shaped dict (file_id, chart_type, title, and
            the right column names for that chart_type), worked out by the Tabular Agent from
            the actual objective and data, not by you guessing from column names. Only build your
            own sections from scratch when visualization_plan was empty.

            Each section is a ChartSpec containing a file_id, chart_type, and the
            required column names:
            - bar/line: label_column + value_column(s), or label_column + series_column + value_column
            - timeline: time_column + value_column(s), or time_column + series_column + value_column
            - scatter3d/surface: x_column, y_column, z_column

            Returns the generated dashboard path."""
        if self.reporting is None:
            raise RuntimeError("no storage configured, cannot generate files")

        resolved_sections = [self._resolve_chart_spec(section) for section in sections]

        if not real_time:
            return self.reporting.generate_dashboard(self.workspace_id, title, resolved_sections, name)

        if not self._last_transform_script:
            raise RuntimeError(
                "generate_dashboard was called with real_time=True, but no invoke_tabular_agent "
                "call this investigation captured a run_python script to make it refreshable. "
                "Call invoke_tabular_agent first (it must run run_python), then generate_dashboard "
                "with real_time=True right after, using file_ids from that same call."
            )

        return self.reporting.generate_realtime_dashboard_bundle(
            self.workspace_id, title, resolved_sections, self._last_transform_script,
            self._last_tabular_file_ids, name,
        )

    def _known_artifact_refs(self) -> list:
        """Every real file_id this investigation has actually produced so far - pulled from
        the TabularFindings/DocumentFindings already recorded on self.state by _record_event,
        never re-derived or guessed. Ground truth for _resolve_file_id's typo correction."""
        refs = []
        for event in self.state.completed_tasks:
            for ref in getattr(event.result, "artifact_refs", None) or []:
                if isinstance(ref, str) and ref not in refs:
                    refs.append(ref)
        return refs

    def _artifact_path(self, file_id: str) -> Optional[str]:
        """Full on-disk path for a file_id under this orchestrator's own workspace, or None if
        it isn't a resolvable id at all (never raises - _resolve_file_id below is what turns
        "doesn't exist" into a helpful error)."""
        if self.storage is None:
            return None
        try:
            return get_parquet_path(self.storage.root_dir, self.workspace_id, file_id)
        except (InvalidArtifactIdError, AttributeError, TypeError):
            return None

    def _resolve_file_id(self, file_id: str) -> str:
        """Resolve and validate a generated artifact file_id.

        Checks that the file_id refers to an existing artifact from the current
        investigation. If needed, automatically corrects minor typos in the file_id.
        Raises an error only if no matching artifact can be found."""
        path = self._artifact_path(file_id)
        if path is not None and self.storage.exists(path):
            return file_id

        known = self._known_artifact_refs()
        if not known:
            return file_id  # nothing to correct against - let the caller's own error surface

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
            # Belt-and-suspenders, not the primary defense anymore: list_files/search_files/the
            # per-turn catalog brief all filter through FileCatalog.is_browsable now, so an xlsx
            # workbook's own file_id (output_ref="" - a workbook has no single "whole file"
            # parquet, see xlsx_ingestor.py) is never shown to the orchestrator to pick in the
            # first place. This still fires for a PDF's main file_id (output_ref is a
            # vector-store pointer, "workspace_{id}", not a real artifact id - same ambiguity
            # worker_service/tasks/investigation.py works around on the ingestion side) - PDFs
            # stay visible/browsable since invoke_document_agent needs that exact file_id, so
            # this is still the first line of defense against handing a PDF's file_id to
            # invoke_tabular_agent by mistake. Also covers any stale file_id reaching here
            # another way (an older chat's thread_context.files_used, etc.) rather than only
            # ones just seen via list_files this turn. Raising here - before a TabularAgent is
            # even created - lets the model self-correct immediately instead of burning a whole
            # invoke_tabular_agent round trip on a guaranteed failure several turns later, deep
            # inside the Docker sandbox.
            raise ValueError(
                f"file_id '{file_ref.file_id}' ('{entry.filename}') has no queryable tabular "
                "data of its own - it's a PDF or xlsx workbook's main entry, not an actual "
                "table. Call list_tables to find the individual table file_id(s) extracted from "
                "it, and pass those to invoke_tabular_agent instead."
            )
        # TabularFileRef only carries file_id/filename now - the real parquet path is derived
        # on demand inside TabularTools via get_parquet_path(root_dir, workspace_id, file_id),
        # never read off the catalog entry as a string.
        return TabularFileRef(file_id=entry.file_id, filename=entry.filename)

    def _get_vector_store(self):
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore()
        return self._vector_store

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker
