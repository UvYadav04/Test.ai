from datetime import datetime

from rapidfuzz import fuzz


class OrchestratorTools:
    def __init__(self, catalog, tabular_agent=None, document_agent=None):
        self.catalog = catalog
        self.tabular_agent = tabular_agent
        self.document_agent = document_agent

    def list_files(self, workspace_id: str, filters: dict = None, max_results: int = 20) -> list:
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
        scored = [(fuzz.partial_ratio(query.lower(), e.filename.lower()), e) for e in self.catalog.all()]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    def get_file_details(self, file_id: str):
        entry = self.catalog.entries.get(file_id)
        if entry is None:
            raise ValueError(f"file_id '{file_id}' not found")
        return entry

    def invoke_tabular_agent(self, objective: str, assigned_files: list, constraints: dict = None):
        if self.tabular_agent is None:
            raise RuntimeError("no tabular agent wired in")
        return self.tabular_agent(objective, assigned_files, constraints or {})

    def invoke_document_agent(self, objective: str, assigned_files: list, constraints: dict = None):
        if self.document_agent is None:
            raise RuntimeError("no document agent wired in")
        return self.document_agent(objective, assigned_files, constraints or {})
