import os
import uuid

import pypdf

from ingestion.file_types.base import BaseIngestor
from ingestion.file_types.pdf.chunker import BaseChunker, DoclingChunker
from ingestion.file_types.pdf.llamaparse_client import parse_pdf_pages
from ingestion.file_types.pdf.utils import extract_tables, is_scanned
from ingestion.models import IngestionResult
from vectordb.schema import ChunkRecord


class PDFIngestor(BaseIngestor):
    def __init__(self, storage=None, vector_store=None, chunker: BaseChunker = None):
        super().__init__(storage=storage, vector_store=vector_store)
        self.chunker = chunker or DoclingChunker()
        self.errors = []
        # parse_pdf_pages() calls LlamaParse's cloud API - a real network request with real
        # latency and per-page cost, so cache by file_path for the same reason the old
        # local-docling conversion cache did: extract_metadata() and ingest() can both be
        # called on this same instance for the same file_path, and without this each call
        # would re-upload the file and burn a second LlamaParse credit for nothing.
        self._pages_cache: dict[str, tuple] = {}

    def _pages_cached(self, file_path: str) -> tuple:
        if file_path not in self._pages_cache:
            self._pages_cache[file_path] = parse_pdf_pages(file_path)
        return self._pages_cache[file_path]

    def validate(self, file_path: str) -> bool:
        """Structural-only check - file exists, is a real parseable PDF with >=1 page.
        Deliberately does NOT call LlamaParse: that's a billed network call, and burning one
        on a file that turns out not to even be a real PDF wastes money/quota for nothing.
        pypdf does a local structural parse only (xref/trailer/page tree - no OCR, no
        network, well under a second)."""
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            self.errors = ["file does not exist or is empty"]
            return False
        try:
            reader = pypdf.PdfReader(file_path)
            if len(reader.pages) == 0:
                self.errors = ["PDF has 0 pages"]
                return False
            self.errors = []
            return True
        except Exception as exc:
            self.errors = [str(exc)]
            return False

    def extract_metadata(self, file_path: str) -> dict:
        pages, _ = self._pages_cached(file_path)
        scanned = all(is_scanned(page_doc) for _, page_doc in pages) if pages else True
        return {"page_count": len(pages), "is_scanned": scanned}

    def ingest(self, file_path: str, workspace_id: str, file_id: str) -> IngestionResult:
        try:
            if self.vector_store is None:
                # Checked before _pages_cached() so a misconfigured ingestor fails fast
                # instead of burning a LlamaParse call it can't do anything useful with.
                raise RuntimeError("no vector store provided")

            pages, errors = self._pages_cached(file_path)
            errors = list(errors)
            # No "PDF looks scanned, OCR not implemented" warning here anymore - unlike the
            # old local docling pipeline (do_ocr=False), LlamaParse OCRs scanned pages itself.

            chunk_records = []
            extracted_tables = []
            table_chunk_records = []
            chunk_index = 0
            table_index = 0

            for page_no, page_doc in pages:
                page_chunks = self.chunker.chunk_document(page_doc)
                for chunk in page_chunks:
                    chunk_records.append(ChunkRecord(
                        chunk_id=f"{file_id}_{chunk_index}_{uuid.uuid4().hex[:8]}",
                        file_id=file_id,
                        workspace_id=workspace_id,
                        text=chunk.text,
                        metadata={"page": page_no, "chunk_index": chunk_index, "section": chunk.section},
                    ))
                    chunk_index += 1

                page_tables, page_table_records, page_candidate_count = self._extract_tables(
                    page_doc, page_chunks, page_no, workspace_id, file_id, errors, table_index,
                )
                # Advance by the CANDIDATE count, not len(page_tables) (the successful-write
                # count) - if a table on this page failed and got skipped (see the try/except
                # below), using the smaller successful count here would let the next page's
                # first table_file_id collide with this page's own surviving table numbering.
                table_index += page_candidate_count
                extracted_tables.extend(page_tables)
                table_chunk_records.extend(page_table_records)

            all_records = chunk_records + table_chunk_records
            if all_records:
                self.vector_store.upsert(all_records)

            status = "success" if not errors else "partial"

            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status=status,
                output_ref=f"workspace_{workspace_id}",
                schema_summary={"page_count": len(pages)},
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

    def _extract_tables(
        self, document, chunks: list, page_no: int, workspace_id: str, file_id: str,
        errors: list, start_index: int,
    ) -> tuple:
        # page_override=page_no: this `document` is a single-LlamaParse-page DoclingDocument
        # (see llamaparse_client.py), so docling's own table provenance would just say "page
        # 1" for every page - page_no is the true page number LlamaParse reported.
        tables = extract_tables(document, chunks, page_override=page_no)
        if not tables:
            return [], [], 0

        if self.storage is None:
            errors.append(f"no storage provided, skipped {len(tables)} table(s)")
            return [], [], len(tables)

        extracted_tables = []
        table_chunk_records = []
        for offset, table in enumerate(tables):
            # start_index carries a running count across pages (each page's own `tables`
            # list restarts at 0) so table_file_id stays globally unique per PDF.
            table_file_id = f"{file_id}_table_{start_index + offset}"
            dataframe = table["dataframe"]
            try:
                output_ref = self.storage.write(dataframe, f"{workspace_id}/{table_file_id}.parquet")
            except Exception as exc:
                # One malformed table (e.g. a financial table whose header collapsed to
                # duplicate/unwriteable column names - dedupe_columns() in storage/base.py
                # now prevents that specific case, but this is defense-in-depth against
                # whatever the next weird table looks like) used to raise straight out of
                # this loop into ingest()'s outer try/except, which discarded EVERYTHING:
                # every already-written table before it and every prose chunk for the whole
                # document, none of which had reached vector_store.upsert() yet. Skipping
                # just this one table keeps the rest of the document's ingestion intact.
                errors.append(f"table on page {table['page']} skipped - failed to write: {exc}")
                continue
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

        return extracted_tables, table_chunk_records, len(tables)
