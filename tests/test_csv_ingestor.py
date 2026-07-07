import pandas as pd
import pytest

from ingestion.file_types.csv.csv_ingestor import CSVIngestor
from ingestion.storage.local_store import LocalParquetStore


@pytest.fixture
def storage(tmp_path):
    return LocalParquetStore(root_dir=str(tmp_path / "parquet"))


@pytest.fixture
def valid_csv(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCarol,40,SF\n")
    return str(path)


def test_validate_accepts_well_formed_csv(valid_csv, storage):
    ingestor = CSVIngestor(storage=storage)
    assert ingestor.validate(valid_csv) is True


def test_validate_rejects_empty_file(tmp_path, storage):
    path = tmp_path / "empty.csv"
    path.write_text("")
    ingestor = CSVIngestor(storage=storage)
    assert ingestor.validate(str(path)) is False


def test_validate_rejects_missing_file(storage):
    ingestor = CSVIngestor(storage=storage)
    assert ingestor.validate("/no/such/file.csv") is False


def test_extract_metadata_returns_columns_and_dtypes(valid_csv, storage):
    ingestor = CSVIngestor(storage=storage)
    meta = ingestor.extract_metadata(valid_csv)
    assert meta["columns"] == ["name", "age", "city"]
    assert "age" in meta["dtypes"]
    assert meta["delimiter"] == ","


def test_ingest_success_writes_parquet_and_returns_result(valid_csv, storage):
    ingestor = CSVIngestor(storage=storage)
    result = ingestor.ingest(valid_csv, workspace_id="ws1", file_id="file1")

    assert result.status == "success"
    assert result.row_count == 3
    assert result.errors == []
    assert storage.exists(result.output_ref)

    df = storage.read(result.output_ref)
    assert list(df.columns) == ["name", "age", "city"]
    assert len(df) == 3


def test_validate_rejects_binary_garbage_file(tmp_path, storage):
    path = tmp_path / "corrupt.csv"
    path.write_bytes(bytes(range(256)) * 4)
    ingestor = CSVIngestor(storage=storage)
    assert ingestor.validate(str(path)) is False


def test_ingest_failure_when_storage_missing_returns_failed_status(valid_csv):
    # No storage backend injected -> ingest() must catch the resulting error
    # and report it as a structured failure, never raise.
    ingestor = CSVIngestor(storage=None)
    result = ingestor.ingest(valid_csv, workspace_id="ws1", file_id="file2")
    assert result.status == "failed"
    assert result.errors
