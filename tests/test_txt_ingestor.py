import pytest

from ingestion.file_types.pdf.chunker import FixedSizeChunker
from ingestion.file_types.txt.txt_ingestor import TXTIngestor
from tests.fakes import FakeVectorStore


@pytest.fixture
def vector_store():
    return FakeVectorStore()


@pytest.fixture
def text_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text(
        "# Meeting notes\n\n"
        "This is a real paragraph of body text. It contains enough characters to look like "
        "genuine content rather than a placeholder.\n\n"
        "## Follow-ups\n\n"
        "A second section with its own paragraph of text to give the chunker something to "
        "split across sections.\n"
    )
    return str(path)


@pytest.fixture
def empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    return str(path)


def test_validate_accepts_real_text_file(text_file, vector_store):
    ingestor = TXTIngestor(vector_store=vector_store)
    assert ingestor.validate(text_file) is True


def test_validate_rejects_empty_file(empty_file, vector_store):
    ingestor = TXTIngestor(vector_store=vector_store)
    assert ingestor.validate(empty_file) is False
    assert ingestor.errors


def test_validate_rejects_missing_file(vector_store):
    ingestor = TXTIngestor(vector_store=vector_store)
    assert ingestor.validate("/no/such/file.txt") is False


def test_extract_metadata_reports_char_count(text_file, vector_store):
    ingestor = TXTIngestor(vector_store=vector_store)
    meta = ingestor.extract_metadata(text_file)
    assert meta["char_count"] > 0


def test_ingest_success_chunks_via_docling_and_upserts_into_vector_store(text_file, vector_store):
    # Default chunker is DoclingChunker (same HybridChunker PDFIngestor uses) - no bespoke
    # text splitter for txt.
    ingestor = TXTIngestor(vector_store=vector_store)
    result = ingestor.ingest(text_file, workspace_id="ws1", file_id="file1")

    assert result.status == "success"
    assert result.chunk_count > 0
    assert result.schema_summary["char_count"] > 0
    assert result.output_ref == "workspace_ws1"

    stored_ids = list(vector_store._store.keys())
    assert len(stored_ids) == result.chunk_count


def test_ingest_accepts_a_swapped_in_chunker(text_file, vector_store):
    # chunker stays pluggable - a caller can still opt into FixedSizeChunker if it ever needs to.
    ingestor = TXTIngestor(vector_store=vector_store, chunker=FixedSizeChunker(chunk_size=80, overlap=10))
    result = ingestor.ingest(text_file, workspace_id="ws1", file_id="file1b")
    assert result.status == "success"
    assert result.chunk_count > 0


def test_ingest_failure_when_vector_store_missing_returns_failed_status(text_file):
    ingestor = TXTIngestor(vector_store=None)
    result = ingestor.ingest(text_file, workspace_id="ws1", file_id="file2")
    assert result.status == "failed"
    assert result.errors


def test_ingest_file_routes_txt_through_txt_ingestor(tmp_path, vector_store):
    from ingestion.manager import IngestionManager
    from ingestion.storage.local_store import LocalParquetStore

    manager = IngestionManager(storage=LocalParquetStore(root_dir=str(tmp_path / "parquet")), vector_store=vector_store)
    path = tmp_path / "data.txt"
    path.write_text("Some plain text content for the manager routing test, long enough to chunk.\n" * 5)

    result = manager.ingest_file(str(path), workspace_id="ws1", file_id="f1")

    assert result.status == "success"
    assert result.chunk_count > 0
