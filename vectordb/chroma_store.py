import chromadb

from vectordb.base import BaseVectorStore
from vectordb.schema import ChunkRecord

EMBEDDING_DIM = 8


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_directory: str = "data/chroma"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="chunks")

    def upsert(self, chunks: list) -> list:
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"file_id": c.file_id, "workspace_id": c.workspace_id, **c.metadata}
            for c in chunks
        ]
        embeddings = [c.embedding if c.embedding else [0.0] * EMBEDDING_DIM for c in chunks]

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return ids

    def query(self, embedding: list, top_k: int, filters: dict = None) -> list:
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k, where=filters)
        return self._to_chunks(result, batched=True)

    def get_by_id(self, ids: list) -> list:
        result = self.collection.get(ids=ids)
        return self._to_chunks(result, batched=False)

    def delete(self, ids: list) -> None:
        self.collection.delete(ids=ids)

    def _to_chunks(self, result: dict, batched: bool) -> list:
        ids = result["ids"][0] if batched else result["ids"]
        documents = result["documents"][0] if batched else result["documents"]
        metadatas = result["metadatas"][0] if batched else result["metadatas"]

        chunks = []
        for i, chunk_id in enumerate(ids):
            meta = dict(metadatas[i])
            file_id = meta.pop("file_id", "")
            workspace_id = meta.pop("workspace_id", "")
            chunks.append(ChunkRecord(
                chunk_id=chunk_id,
                file_id=file_id,
                workspace_id=workspace_id,
                text=documents[i],
                metadata=meta,
            ))
        return chunks
