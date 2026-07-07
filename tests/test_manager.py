import json

import pytest

from ingestion.manager import IngestionManager
from ingestion.storage.local_store import LocalParquetStore
from tests.fakes import FakeVectorStore


@pytest.fixture
def manager(tmp_path):
    storage = LocalParquetStore(root_dir=str(tmp_path / "parquet"))
    vector_store = FakeVectorStore()
    return IngestionManager(storage=storage, vector_store=vector_store)


def test_ingest_file_routes_csv_through_csv_ingestor(tmp_path, manager):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4\n")

    result = manager.ingest_file(str(path), workspace_id="ws1", file_id="f1")

    assert result.status == "success"
    assert result.row_count == 2
    assert manager.storage.exists(result.output_ref)


def test_ingest_file_routes_json_through_json_ingestor(tmp_path, manager):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}]))

    result = manager.ingest_file(str(path), workspace_id="ws1", file_id="f2")

    assert result.status == "success"
    assert result.row_count == 2


def test_ingest_file_unsupported_extension_returns_failed(tmp_path, manager):
    path = tmp_path / "data.xyz"
    path.write_text("whatever")

    result = manager.ingest_file(str(path), workspace_id="ws1", file_id="f3")

    assert result.status == "failed"
    assert "Unsupported file type" in result.errors[0]


def test_ingest_file_invalid_file_returns_failed_via_validate(tmp_path, manager):
    path = tmp_path / "empty.csv"
    path.write_text("")

    result = manager.ingest_file(str(path), workspace_id="ws1", file_id="f4")

    assert result.status == "failed"
    assert result.errors == ["validation failed"]


def test_ingest_file_missing_file_returns_failed(manager):
    result = manager.ingest_file("/no/such/file.csv", workspace_id="ws1", file_id="f5")
    assert result.status == "failed"
