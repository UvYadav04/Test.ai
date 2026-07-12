from ingestion import registry
from ingestion.models import IngestionResult
from ingestion.storage.base import BaseObjectStore
from vectordb.base import BaseVectorStore


class IngestionManager:
    def __init__(self, storage: BaseObjectStore, vector_store: BaseVectorStore):
        self.storage = storage
        self.vector_store = vector_store

    def ingest_file(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        try:
            ingestor_cls = registry.get_ingestor_for(file_path)
        except ValueError as exc:
            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="failed",
                output_ref="",
                schema_summary={},
                errors=[str(exc)],
            )

        ingestor = ingestor_cls(storage=self.storage, vector_store=self.vector_store)

        if not ingestor.validate(file_path):
            print("validation error")
            errors = getattr(ingestor, "errors", None) or ["validation failed"]
            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="failed",
                output_ref="",
                schema_summary={},
                errors=errors,
            )
        
        return ingestor.ingest(file_path, workspace_id, file_id)
