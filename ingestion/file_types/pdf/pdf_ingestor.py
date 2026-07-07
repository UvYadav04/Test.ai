import uuid

from ingestion.file_types.base import BaseIngestor
from ingestion.file_types.pdf.chunker import BaseChunker, SemanticChunker
from ingestion.file_types.pdf.utils import extract_text_per_page, is_scanned
from ingestion.models import IngestionResult
from vectordb.schema import ChunkRecord


class PDFIngestor(BaseIngestor):
    def __init__(self, storage=None, vector_store=None, chunker: BaseChunker = None):
        super().__init__(storage=storage, vector_store=vector_store)
        self.chunker = chunker or SemanticChunker()

    def validate(self, file_path: str) -> bool:
        try:
            pages = extract_text_per_page(file_path)
            return len(pages) > 0
        except Exception:
            return False

    def extract_metadata(self, file_path: str) -> dict:
        pages = extract_text_per_page(file_path)
        return {"page_count": len(pages), "is_scanned": is_scanned(pages)}

    def ingest(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        try:
            pages = extract_text_per_page(file_path)
            errors = []

            if is_scanned(pages):
                errors.append("PDF looks scanned, OCR not implemented yet")

            chunks = self.chunker.chunk_pages(pages)

            chunk_records = []
            for chunk in chunks:
                chunk_records.append(ChunkRecord(
                    chunk_id=f"{file_id}_{chunk.chunk_index}_{uuid.uuid4().hex[:8]}",
                    file_id=file_id,
                    workspace_id=workspace_id,
                    text=chunk.text,
                    embedding=None,
                    metadata={"page": chunk.page, "chunk_index": chunk.chunk_index},
                ))

            if self.vector_store is None:
                raise RuntimeError("no vector store provided")

            if chunk_records:
                self.vector_store.upsert(chunk_records)

            status = "success" if not errors else "partial"

            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status=status,
                output_ref=f"workspace_{workspace_id}",
                schema_summary={"page_count": len(pages)},
                chunk_count=len(chunk_records),
                errors=errors,
            )
        except Exception as exc:
            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="failed",
                output_ref="",
                schema_summary={},
                errors=[str(exc)],
            )
