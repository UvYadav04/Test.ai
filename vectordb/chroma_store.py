import chromadb

from config import get_settings
from vectordb.base import BaseVectorStore
from vectordb.schema import ChunkRecord


class ChromaVectorStore(BaseVectorStore):
    def __init__(self):
        settings = get_settings()
        self.client = chromadb.CloudClient(
            tenant=settings.CHROMA_TENANT,
            database=settings.CHROMA_DATABASE,
            api_key=settings.CHROMA_API_KEY,
        )
        self.collection = self.client.get_or_create_collection(name="chunks")

    def upsert(self, chunks: list) -> list:
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"file_id": c.file_id, "workspace_id": c.workspace_id, **c.metadata}
            for c in chunks
        ]

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return ids

    def query(self, query_text: str, top_k: int, filters: dict = None) -> list:
        result = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_chunks(result, batched=True)

    def get_by_id(self, ids: list) -> list:
        result = self.collection.get(ids=ids)
        return self._to_chunks(result, batched=False)

    def get_by_filter(self, filters: dict) -> list:
        result = self.collection.get(where=filters)
        return self._to_chunks(result, batched=False)

    def delete(self, ids: list) -> None:
        self.collection.delete(ids=ids)

    def _to_chunks(self, result: dict, batched: bool) -> list:
        ids = result["ids"][0] if batched else result["ids"]
        documents = result["documents"][0] if batched else result["documents"]
        metadatas = result["metadatas"][0] if batched else result["metadatas"]
        distances = result.get("distances")
        distances = (distances[0] if batched else distances) if distances else None

        chunks = []
        for i, chunk_id in enumerate(ids):
            meta = dict(metadatas[i])
            file_id = meta.pop("file_id", "")
            workspace_id = meta.pop("workspace_id", "")
            if distances:
                meta["score"] = distances[i]
            chunks.append(ChunkRecord(
                chunk_id=chunk_id,
                file_id=file_id,
                workspace_id=workspace_id,
                text=documents[i],
                metadata=meta,
            ))
        return chunks
