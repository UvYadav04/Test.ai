import os

import pytest
from reportlab.pdfgen import canvas

from ingestion.file_types.pdf.chunker import FixedSizeChunker
from ingestion.file_types.pdf.pdf_ingestor import PDFIngestor
from tests.fakes import FakeVectorStore

# extract_metadata()/ingest() now call LlamaParse's cloud API (see llamaparse_client.py) -
# validate() stays offline (pypdf-only), but these two need a real key and network access,
# so they're skipped rather than failing/hanging in CI or an offline dev environment.
requires_llamaparse = pytest.mark.skipif(
    not os.getenv("LLAMAPARSE_API_KEY"),
    reason="requires LLAMAPARSE_API_KEY (PDFIngestor parses via LlamaParse's cloud API)",
)


@pytest.fixture
def vector_store():
    return FakeVectorStore()


@pytest.fixture
def text_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    c = canvas.Canvas(str(path))
    for page_num in range(2):
        c.drawString(72, 700, f"Page {page_num + 1}: This is a real paragraph of body text.")
        c.drawString(72, 680, "It contains enough characters to look like a genuine text PDF.")
        c.showPage()
    c.save()
    return str(path)


@pytest.fixture
def corrupt_pdf(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a real pdf body\n%%EOF")
    return str(path)


def test_validate_accepts_real_pdf(text_pdf, vector_store):
    ingestor = PDFIngestor(vector_store=vector_store)
    assert ingestor.validate(text_pdf) is True


def test_validate_rejects_corrupt_pdf(corrupt_pdf, vector_store):
    ingestor = PDFIngestor(vector_store=vector_store)
    assert ingestor.validate(corrupt_pdf) is False


@requires_llamaparse
def test_extract_metadata_reports_page_count_and_scanned_flag(text_pdf, vector_store):
    ingestor = PDFIngestor(vector_store=vector_store)
    meta = ingestor.extract_metadata(text_pdf)
    assert meta["page_count"] == 2
    assert meta["is_scanned"] is False


@requires_llamaparse
def test_ingest_success_chunks_and_upserts_into_vector_store(text_pdf, vector_store):
    ingestor = PDFIngestor(vector_store=vector_store, chunker=FixedSizeChunker(chunk_size=50, overlap=10))
    result = ingestor.ingest(text_pdf, workspace_id="ws1", file_id="file1")

    assert result.status == "success"
    assert result.chunk_count > 0
    assert result.schema_summary["page_count"] == 2
    assert result.output_ref == vector_store.collection_name_for("ws1")

    stored_ids = list(vector_store._store.keys())
    assert len(stored_ids) == result.chunk_count
    # Embeddings are intentionally left unset at ingestion time.
    assert all(c.embedding is None for c in vector_store._store.values())


def test_ingest_failure_when_vector_store_missing_returns_failed_status(text_pdf):
    ingestor = PDFIngestor(vector_store=None)
    result = ingestor.ingest(text_pdf, workspace_id="ws1", file_id="file2")
    assert result.status == "failed"
    assert result.errors
