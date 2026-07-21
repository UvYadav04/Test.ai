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
    def __init__(
        self, catalog, state, vector_store=None, reranker=None, memory=None, storage=None,
        reports_dir: str = "data/reports",
    ):
        self.catalog = catalog
        self.state = state
        self.storage = storage
        self.workspace_id = "default"
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
        """List files in the workspace matching a structured filter: name_contains, file_type
        (list, e.g. ["csv","pdf","table"]), uploaded_after/uploaded_before (ISO dates - call
        get_current_date first to resolve relative phrases like "4 months ago"), min_rows,
        max_rows, tags. Combine filters in one call rather than making several narrow calls.

        Only shows files with queryable data of their own: a PDF's/xlsx workbook's own file_id
        never appears here if it has no directly usable data (see FileCatalog.is_browsable) -
        an xlsx workbook's individual sheet tables (file_type "table") show up instead, since
        those are its only queryable surface. A PDF's own per-page tables are deliberately NOT
        listed here even though they exist in the catalog - call invoke_document_agent on the
        PDF itself and use the table_ref it reports back, rather than list_tables, to find
        those."""
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
        """List every xlsx sheet-table in the workspace catalog, each with its OWN file_id you
        can pass to invoke_tabular_agent. An xlsx workbook's own file_id (from list_files/the
        catalog) has no queryable data of its own - a workbook has no single "whole file" table,
        only its individual sheets do - so call this before invoke_tabular_agent whenever the
        file you want to analyze is an xlsx workbook, and use the table file_id(s) this returns
        instead of the workbook's own file_id.

        This does NOT include PDF tables (unlike a plain file_type filter would) - those aren't
        pre-listed anywhere; call invoke_document_agent on the PDF itself and use the table_ref
        it reports back in its findings when it finds one relevant to your objective.
        Shorthand for list_files(filters={"file_type": ["table"]})."""
        return self.list_files(workspace_id, filters={"file_type": ["table"]}, max_results=max_results)

    def list_file_formats(self, workspace_id: str) -> list:
        """List the distinct file types present in the workspace (e.g. ["csv", "pdf",
        "table"]). Use this to see what kinds of data exist before deciding how to filter
        list_files, especially early in an investigation. Same visibility rules as list_files -
        e.g. "table" only appears here if an xlsx workbook actually has browsable sheet-tables,
        never solely because a PDF has (hidden) per-page tables."""
        return sorted({e.file_type for e in self.catalog.browsable()})

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
        assigned_files (each {file_id} - just the file_id, the orchestrator resolves the real
        output_ref from the catalog itself, never trust or invent one). It runs its own
        sandboxed Python/DuckDB tool-calling loop in an isolated context and returns one compact
        TabularFindings - you
        never see its raw code, intermediate output, or the underlying data. Use for CSV/table
        data: aggregates, filters, joins, computed answers - including tables surfaced by the
        Document Agent via table_ref.

        Set must_export=True whenever the result needs to persist afterward (the user asked for
        a CSV, dashboard, or report) - this is enforced independently of how you word `objective`,
        so it survives even if you have to reword the objective on a retry after a failure. When
        True, this call raises instead of returning a fabricated or missing output_ref, so you
        know to retry rather than silently passing a fake reference on to generate_csv/
        generate_dashboard."""
        constraints = constraints or {}
        self.state.selected_files.extend(f.file_id for f in assigned_files)
        tabular_files = [self._to_tabular_file_ref(f) for f in assigned_files]
        agent = TabularAgent(tabular_files, storage=self.storage, workspace_id=self.workspace_id)

        effective_objective = objective
        if must_export:
            effective_objective += (
                "\n\nThis result MUST be persisted: your final computation must call "
                "save(df, name) inside run_python, and you must report its real output_ref "
                "string in your findings' artifact_refs."
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
            valid_refs = [
                ref for ref in result.artifact_refs
                if isinstance(ref, str) and (".parquet" in ref or "/" in ref or "\\" in ref)
            ]
            if not valid_refs:
                raise RuntimeError(
                    "invoke_tabular_agent was called with must_export=True but the Tabular "
                    "Agent did not return a real output_ref (it likely never called save(), or "
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
        """For complex or "why"-style objectives, generate and prioritize candidate
        explanations BEFORE delegating, so investigation effort targets the most likely
        directions first instead of exploring blindly. context = {"available_files": [{"file_id":
        ..., "filename": ..., "file_type": ...}, ... one dict per file, using exactly those
        three field names - not "name"/"type" or any other renaming], "known_findings": [...
        optional, only if this is a follow-up round ...]}. Skip this for simple, direct
        questions - go straight to invoke_tabular_agent/invoke_document_agent instead."""
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
        return self.reporting.generate_csv(self._resolve_output_ref(output_ref), name)

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
        """Build a dashboard with charts from one or more existing data artifacts (output_refs
        from table_refs or a persisted run_python save() call). Use this when the user asks for
        a dashboard or visualization, not a CSV or written report.

        Each item in `sections` is a ChartSpec: {output_ref, chart_type, ...column names...}.
        You never pass or see actual data values here - only an output_ref (a file path) and
        column names you already know from a Tabular Agent's findings; the real numbers are
        read straight from the parquet file when the dashboard is built.

        chart_type options and which column names each needs:
        - "bar" / "line": EITHER label_column + value_columns (1+ numeric series - if omitted,
          the first non-numeric column and up to 5 numeric columns are used automatically) OR,
          when the result has TWO grouping columns and one metric (e.g. Age, Gender, Customer
          Count), label_column + series_column + value_column - this produces one bar/line per
          distinct series_column value grouped along label_column. Use this whenever a Tabular
          Agent's result has more than one grouping column - never pass just one grouping
          column as label_column and silently drop the other.
        - "timeline": time_column (required) plus EITHER value_columns (wide data - one series
          per column) OR series_column + value_column (long/tidy data - one series per distinct
          value in series_column, e.g. columns date, job_title, count ->
          series_column="job_title", value_column="count").
        - "scatter3d" / "surface": x_column, y_column, z_column (all required). "surface" needs
          every (x, y) combination present in the data to build a valid grid - use "scatter3d"
          instead if that can't be guaranteed.

        Set real_time=True when the user wants a dashboard that stays live as their data
        changes (they said something like "keep this updated" or "refresh when I upload new
        data"), not just a one-off snapshot. This does NOT require you to pass a script or
        column data yourself - it automatically reuses the exact code your most recent
        invoke_tabular_agent call ran, so every section in `sections` MUST have come from that
        SAME invoke_tabular_agent call (its save() calls are what generated their output_refs).
        If you need real_time=True but haven't called invoke_tabular_agent yet this
        investigation, or your last one didn't call run_python, this raises instead of silently
        producing a dashboard that can never refresh. Giving each section a `name` (a short
        stable label like "revenue_by_region") is recommended but not required - it falls back
        to a slug of title/index.

        Creates a new folder under today's date named after `name` (falls back to a slug of
        title) and writes the dashboard there together with copies of every source data file
        that fed it. Returns the file path - report it in your final answer and in
        artifact_refs."""
        if self.reporting is None:
            raise RuntimeError("no storage configured, cannot generate files")

        resolved_sections = [self._resolve_chart_spec(section) for section in sections]

        if not real_time:
            return self.reporting.generate_dashboard(title, resolved_sections, name)

        if not self._last_transform_script:
            raise RuntimeError(
                "generate_dashboard was called with real_time=True, but no invoke_tabular_agent "
                "call this investigation captured a run_python script to make it refreshable. "
                "Call invoke_tabular_agent first (it must run run_python), then generate_dashboard "
                "with real_time=True right after, using output_refs from that same call."
            )

        return self.reporting.generate_realtime_dashboard_bundle(
            title, resolved_sections, self._last_transform_script, self._last_tabular_file_ids, name,
        )

    def _known_artifact_refs(self) -> list:
        """Every real output_ref this investigation has actually produced so far - pulled from
        the TabularFindings/DocumentFindings already recorded on self.state by _record_event,
        never re-derived or guessed. Ground truth for _resolve_output_ref's typo correction."""
        refs = []
        for event in self.state.completed_tasks:
            for ref in getattr(event.result, "artifact_refs", None) or []:
                if isinstance(ref, str) and ref not in refs:
                    refs.append(ref)
        return refs

    def _resolve_output_ref(self, output_ref: str) -> str:
        """Guards generate_csv/generate_dashboard's output_ref argument against a failure mode
        that's bitten in practice: to call either tool, the orchestrator LLM has to retype a
        long, effectively-random parquet path (a workspace-id hex string plus a result-id hex
        suffix, ~40+ characters) from its own context into a fresh tool call - unlike file_id
        lookups elsewhere in this class, there's no server-side table to validate an output_ref
        against, so a single mistyped character produces a path that looks right but points at
        nothing (see the "[Errno 2] No such file or directory" failures this fixes).

        If the exact string isn't on disk, fuzzy-match it against every real output_ref this
        investigation has actually produced (self._known_artifact_refs) - the same rapidfuzz
        tool search_files already uses for "the model's string doesn't exactly match reality" -
        and silently correct a near-miss rather than failing the whole investigation turn over
        one wrong character. Raises (listing the real options, so the model can retry with an
        exact copy instead of guessing again) only when nothing is close enough."""
        if self.storage is not None and self.storage.exists(output_ref):
            return output_ref

        known = self._known_artifact_refs()
        if not known:
            return output_ref  # nothing to correct against - let the caller's own error surface

        best_ref, best_score = None, 0.0
        for ref in known:
            score = fuzz.ratio(output_ref, ref)
            if score > best_score:
                best_ref, best_score = ref, score

        if best_ref is not None and best_score >= 90:
            return best_ref

        raise ValueError(
            f"output_ref '{output_ref}' does not exist and doesn't closely match any artifact "
            f"this investigation has actually produced. Real output_refs so far: {known}. Copy "
            "one of these exactly rather than retyping it."
        )

    def _resolve_chart_spec(self, raw) -> ChartSpec:
        spec = ReportingTools._to_chart_spec(raw)
        spec.output_ref = self._resolve_output_ref(spec.output_ref)
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
        if not (entry.output_ref or "").endswith(".parquet"):
            # Belt-and-suspenders, not the primary defense anymore: list_files/search_files/the
            # per-turn catalog brief all filter through FileCatalog.is_browsable now, so an xlsx
            # workbook's own file_id (output_ref="" - a workbook has no single "whole file"
            # parquet, see xlsx_ingestor.py) is never shown to the orchestrator to pick in the
            # first place. This still fires for a PDF's main file_id (output_ref is a
            # vector-store pointer, "workspace_{id}", not a filesystem path - same ambiguity
            # worker_service/tasks/investigation.py's _looks_like_local_parquet_ref works around
            # on the ingestion side) - PDFs stay visible/browsable since invoke_document_agent
            # needs that exact file_id, so this is still the first line of defense against
            # handing a PDF's file_id to invoke_tabular_agent by mistake. Also covers any stale
            # file_id reaching here another way (an older chat's thread_context.files_used,
            # etc.) rather than only ones just seen via list_files this turn. Raising here -
            # before a TabularAgent is even created - lets the model self-correct immediately
            # instead of burning a whole invoke_tabular_agent round trip on a guaranteed failure
            # several turns later, deep inside the Docker sandbox.
            raise ValueError(
                f"file_id '{file_ref.file_id}' ('{entry.filename}') has no queryable tabular "
                "data of its own - it's a PDF or xlsx workbook's main entry, not an actual "
                "table. Call list_tables to find the individual table file_id(s) extracted from "
                "it, and pass those to invoke_tabular_agent instead."
            )
        return TabularFileRef(file_id=entry.file_id, output_ref=entry.output_ref, filename=entry.filename)

    def _get_vector_store(self):
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore()
        return self._vector_store

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker
