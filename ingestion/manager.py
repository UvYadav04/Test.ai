import logging

from ingestion import registry
from ingestion.models import IngestionResult
from ingestion.storage.base import BaseObjectStore
from vectordb.base import BaseVectorStore

logger = logging.getLogger("ingestion.manager")


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
            errors = getattr(ingestor, "errors", None) or ["validation failed"]
            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="failed",
                output_ref="",
                schema_summary={},
                errors=errors,
            )

        # Unlike the two branches above, ingestor.ingest() itself isn't guaranteed to always
        # return cleanly - a corrupt file, an out-of-memory condition on an oversized PDF/txt, or
        # any other unhandled exception deep in a specific ingestor previously propagated straight
        # out of ingest_file(), past run_ingestion's own try/except (which only wraps the
        # download + this call together, not this call's internals specifically), and left the
        # File doc stuck at status="processing" forever instead of ever being marked failed. This
        # keeps ingest_file's contract ("never raises, always returns an IngestionResult") honest.
        try:
            return ingestor.ingest(file_path, workspace_id, file_id)
        except Exception as exc:
            logger.exception("ingest_file: %s ingestor raised for file %s", ingestor_cls.__name__, file_id)
            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="failed",
                output_ref="",
                schema_summary={},
                errors=[f"Unexpected error during ingestion: {exc}"],
            )
