from datetime import datetime, timezone
from typing import Optional

from rapidfuzz import fuzz

from agents.document import DocumentAgent
from agents.tabular import TabularAgent
from tools.hypothesis.hypothesis_tools import HypothesisTools
from tools.hypothesis.models import HypothesisResult
from tools.orchestrator.memory import LongTermMemory
from tools.orchestrator.models import FileRef, InvestigationEvent
from tools.tabular.models import FileRef as TabularFileRef
from vectordb.chroma_store import ChromaVectorStore
from vectordb.reranker import CrossEncoderReranker


class OrchestratorTools:
    def __init__(self, catalog, state, vector_store=None, reranker=None, memory=None):
        self.catalog = catalog
        self.state = state
        self._vector_store = vector_store
        self._reranker = reranker
        self.memory = memory or LongTermMemory()
        self.hypothesis_tools = HypothesisTools()

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

    async def invoke_tabular_agent(self, objective: str, assigned_files: list[FileRef], constraints: Optional[dict] = None):
        """Delegate a data-analysis question to the Tabular Agent, scoped only to the given
        assigned_files (each {file_id, output_ref}). It runs its own DuckDB tool-calling loop
        in an isolated context and returns one compact TabularFindings - you never see its raw
        queries or intermediate results. Use for CSV/table data: aggregates, filters, joins,
        computed answers - including tables surfaced by the Document Agent via table_ref."""
        constraints = constraints or {}
        tabular_files = [self._to_tabular_file_ref(f) for f in assigned_files]
        agent = TabularAgent(tabular_files)
        result = await agent.run(objective, constraints)
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
