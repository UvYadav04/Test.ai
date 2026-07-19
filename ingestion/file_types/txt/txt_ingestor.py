import uuid

from ingestion.file_types.base import BaseIngestor
from ingestion.file_types.pdf.chunker import BaseChunker, DoclingChunker
from ingestion.file_types.txt.utils import convert_document
from ingestion.models import IngestionResult
from vectordb.schema import ChunkRecord


class TXTIngestor(BaseIngestor):
    """Plain text: no tables, no OCR - convert through docling (auto-detected as markdown,
    see txt/utils.py) and chunk with the same DoclingChunker/HybridChunker PDFIngestor uses,
    rather than a bespoke splitter, so section-aware chunking behaves the same way across both
    formats. Mirrors PDFIngestor's shape minus everything PDF-structure-specific (pipeline
    options, table extraction, scanned-page detection - plain text can't be "scanned")."""

    def __init__(self, storage=None, vector_store=None, chunker: BaseChunker = None):
        super().__init__(storage=storage, vector_store=vector_store)
        self.chunker = chunker or DoclingChunker()
        self.errors = []

    def validate(self, file_path: str) -> bool:
        try:
            document, errors = convert_document(file_path)
            self.errors = errors
            if not document.export_to_text().strip():
                self.errors = self.errors or ["file contains no readable text"]
                return False
            return True
        except Exception as exc:
            self.errors = [str(exc)]
            return False

    def extract_metadata(self, file_path: str) -> dict:
        document, _ = convert_document(file_path)
        return {"char_count": len(document.export_to_text())}

    def ingest(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        try:
            document, errors = convert_document(file_path)

            if self.vector_store is None:
                raise RuntimeError("no vector store provided")

            chunks = self.chunker.chunk_document(document)
            chunk_records = [
                ChunkRecord(
                    chunk_id=f"{file_id}_{chunk.chunk_index}_{uuid.uuid4().hex[:8]}",
                    file_id=file_id,
                    workspace_id=workspace_id,
                    text=chunk.text,
                    metadata={"page": chunk.page, "chunk_index": chunk.chunk_index, "section": chunk.section},
                )
                for chunk in chunks
            ]

            if chunk_records:
                self.vector_store.upsert(chunk_records)

            status = "success" if not errors else "partial"

            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status=status,
                output_ref=f"workspace_{workspace_id}",
                schema_summary={"char_count": len(document.export_to_text())},
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
