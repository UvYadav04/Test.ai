import json

import pytest

from ingestion.file_types.json.json_ingestor import JSONIngestor
from ingestion.storage.local_store import LocalParquetStore


@pytest.fixture
def storage(tmp_path):
    return LocalParquetStore(root_dir=str(tmp_path / "parquet"))


@pytest.fixture
def flat_json(tmp_path):
    path = tmp_path / "flat.json"
    data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def nested_json(tmp_path):
    path = tmp_path / "nested.json"
    data = [
        {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}, "tags": ["vip", "new"]},
        {"name": "Bob", "address": {"city": "LA", "zip": "90001"}, "tags": ["new"]},
    ]
    path.write_text(json.dumps(data))
    return str(path)


def test_validate_accepts_well_formed_json(flat_json, storage):
    ingestor = JSONIngestor(storage=storage)
    assert ingestor.validate(flat_json) is True


def test_validate_rejects_malformed_json(tmp_path, storage):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json,,,")
    ingestor = JSONIngestor(storage=storage)
    assert ingestor.validate(str(path)) is False


def test_validate_rejects_empty_file(tmp_path, storage):
    path = tmp_path / "empty.json"
    path.write_text("")
    ingestor = JSONIngestor(storage=storage)
    assert ingestor.validate(str(path)) is False


def test_extract_metadata_flat_json(flat_json, storage):
    ingestor = JSONIngestor(storage=storage)
    meta = ingestor.extract_metadata(flat_json)
    assert set(meta["columns"]) == {"name", "age"}
    assert meta["top_level_type"] == "list"


def test_extract_metadata_nested_json_flattens_columns(nested_json, storage):
    ingestor = JSONIngestor(storage=storage)
    meta = ingestor.extract_metadata(nested_json)
    assert "address.city" in meta["columns"]
    assert "address.zip" in meta["columns"]


def test_ingest_success_flat_json_writes_parquet(flat_json, storage):
    ingestor = JSONIngestor(storage=storage)
    result = ingestor.ingest(flat_json, workspace_id="ws1", file_id="file1")

    assert result.status == "success"
    assert result.row_count == 2
    assert storage.exists(result.output_ref)

    df = storage.read(result.output_ref)
    assert set(df.columns) == {"name", "age"}


def test_ingest_success_nested_json_flattens_and_stores_list_columns(nested_json, storage):
    ingestor = JSONIngestor(storage=storage)
    result = ingestor.ingest(nested_json, workspace_id="ws1", file_id="file2")

    assert result.status == "success"
    df = storage.read(result.output_ref)
    assert "address.city" in df.columns
    # `tags` is a list column -- must be JSON-stringified so Parquet can store it.
    assert df["tags"].apply(lambda v: isinstance(v, str)).all()


def test_ingest_failure_when_storage_missing_returns_failed_status(flat_json):
    ingestor = JSONIngestor(storage=None)
    result = ingestor.ingest(flat_json, workspace_id="ws1", file_id="file3")
    assert result.status == "failed"
    assert result.errors
