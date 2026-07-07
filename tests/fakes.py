"""Lightweight in-memory fakes used across tests so ingestor/manager tests
don't require a real Chroma/persistent backend to run fast and in isolation."""
from vectordb.base import BaseVectorStore
from vectordb.schema import ChunkRecord


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._store: dict[str, ChunkRecord] = {}

    def collection_name_for(self, workspace_id: str) -> str:
        return f"fake_{workspace_id}"

    def upsert(self, chunks: list[ChunkRecord]) -> list[str]:
        for chunk in chunks:
            self._store[chunk.chunk_id] = chunk
        return [c.chunk_id for c in chunks]

    def query(self, embedding, top_k: int, filters: dict = None) -> list[ChunkRecord]:
        results = list(self._store.values())
        if filters:
            for key, value in filters.items():
                results = [c for c in results if c.metadata.get(key, getattr(c, key, None)) == value]
        return results[:top_k]

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._store.pop(i, None)

    def get_by_id(self, ids: list[str]) -> list[ChunkRecord]:
        return [self._store[i] for i in ids if i in self._store]
