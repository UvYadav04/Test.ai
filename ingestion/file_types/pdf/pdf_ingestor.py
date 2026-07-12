import uuid

from ingestion.file_types.base import BaseIngestor
from ingestion.file_types.pdf.chunker import BaseChunker, DoclingChunker
from ingestion.file_types.pdf.utils import convert_document, extract_tables, get_page_count, is_scanned
from ingestion.models import IngestionResult
from vectordb.schema import ChunkRecord


class PDFIngestor(BaseIngestor):
    def __init__(self, storage=None, vector_store=None, chunker: BaseChunker = None):
        super().__init__(storage=storage, vector_store=vector_store)
        self.chunker = chunker or DoclingChunker()
        self.errors = []

    def validate(self, file_path: str) -> bool:
        try:
            print("validating pdf...")
            document, errors = convert_document(file_path)
            self.errors = errors
            if get_page_count(document) == 0:
                self.errors = self.errors or ["docling produced a document with 0 pages"]
                return False
            return True
        except Exception as exc:
            print("exception in validation")
            self.errors = [str(exc)]
            print(self.errors)
            return False

    def extract_metadata(self, file_path: str) -> dict:
        document, _ = convert_document(file_path)
        return {"page_count": get_page_count(document), "is_scanned": is_scanned(document)}

    def ingest(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        try:
            document, errors = convert_document(file_path)

            if is_scanned(document):
                errors.append("PDF looks scanned, OCR not implemented yet")

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

            extracted_tables, table_chunk_records = self._extract_tables(document, chunks, workspace_id, file_id, errors)

            all_records = chunk_records + table_chunk_records
            if all_records:
                self.vector_store.upsert(all_records)

            status = "success" if not errors else "partial"

            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status=status,
                output_ref=f"workspace_{workspace_id}",
                schema_summary={"page_count": get_page_count(document)},
                chunk_count=len(chunk_records),
                extracted_tables=extracted_tables,
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

    def _extract_tables(self, document, chunks: list, workspace_id: str, file_id: str, errors: list) -> tuple:
        tables = extract_tables(document, chunks)
        if not tables:
            return [], []

        if self.storage is None:
            errors.append(f"no storage provided, skipped {len(tables)} table(s)")
            return [], []

        extracted_tables = []
        table_chunk_records = []
        for table in tables:
            table_file_id = f"{file_id}_table_{table['index']}"
            dataframe = table["dataframe"]
            output_ref = self.storage.write(dataframe, f"{workspace_id}/{table_file_id}.parquet")
            columns = [str(c) for c in dataframe.columns]
            row_count = len(dataframe)

            extracted_tables.append({
                "file_id": table_file_id,
                "output_ref": output_ref,
                "page": table["page"],
                "row_count": row_count,
                "columns": columns,
            })

            caption = table["caption"]
            text = f"{caption}\nColumns: {', '.join(columns)}"
            table_chunk_records.append(ChunkRecord(
                chunk_id=f"{table_file_id}_{uuid.uuid4().hex[:8]}",
                file_id=file_id,
                workspace_id=workspace_id,
                text=text,
                metadata={
                    "page": table["page"],
                    "type": "table",
                    "table_ref": table_file_id,
                    "row_count": row_count,
                    "columns": ", ".join(columns),
                },
            ))

        return extracted_tables, table_chunk_records
