from llm_provider import LLMProvider
from tools.document.models import ChunkResult, ComparisonResult, SectionInfo, VerificationResult
from tools.llm_call import ask_llm

VERIFY_PROMPT = """Chunk text:
{text}

Claim: {claim}

Does the chunk text support the claim? Answer with SUPPORTED or NOT_SUPPORTED, then a pipe |, then a one sentence reason.
"""


class DocumentTools:
    def __init__(self, assigned_files: list, vector_store, embed_fn, reranker=None, llm_provider=None):
        self.assigned_file_ids = [f.file_id for f in assigned_files]
        self.vector_store = vector_store
        self.embed_fn = embed_fn
        self.reranker = reranker
        self.llm_provider = llm_provider or LLMProvider()

    def _scoped_file_ids(self, file_ids: list = None) -> list:
        if file_ids is None:
            return self.assigned_file_ids
        return [f for f in file_ids if f in self.assigned_file_ids]

    def _to_result(self, chunk) -> ChunkResult:
        meta = dict(chunk.metadata)
        score = meta.pop("score", 0.0)
        return ChunkResult(chunk_id=chunk.chunk_id, file_id=chunk.file_id, text=chunk.text, score=score, metadata=meta)

    def search_documents(self, query: str, file_ids: list = None, top_k: int = 8) -> list:
        scoped = self._scoped_file_ids(file_ids)
        embedding = self.embed_fn(query)
        filters = {"file_id": {"$in": scoped}}
        fetch_k = top_k * 3 if self.reranker else top_k

        chunks = self.vector_store.query(embedding, fetch_k, filters=filters)
        if self.reranker:
            chunks = self.reranker.rank(query, chunks, top_k=top_k)

        return [self._to_result(c) for c in chunks[:top_k]]

    def search_within_file(self, file_id: str, query: str, top_k: int = 8) -> list:
        return self.search_documents(query, file_ids=[file_id], top_k=top_k)

    def get_chunk(self, chunk_id: str) -> ChunkResult:
        results = self.vector_store.get_by_id([chunk_id])
        if not results:
            raise ValueError(f"chunk_id '{chunk_id}' not found")
        return self._to_result(results[0])

    def get_surrounding_chunks(self, chunk_id: str, window: int = 1) -> list:
        target = self.get_chunk(chunk_id)
        chunk_index = target.metadata.get("chunk_index")
        if chunk_index is None:
            return [target]

        filters = {
            "$and": [
                {"file_id": target.file_id},
                {"chunk_index": {"$gte": chunk_index - window}},
                {"chunk_index": {"$lte": chunk_index + window}},
            ]
        }
        chunks = self.vector_store.get_by_filter(filters)
        chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
        return [self._to_result(c) for c in chunks]

    def list_file_sections(self, file_id: str) -> list:
        chunks = self.vector_store.get_by_filter({"file_id": file_id})

        sections = {}
        for chunk in chunks:
            title = chunk.metadata.get("section", "")
            if not title:
                continue
            page = chunk.metadata.get("page", 0)
            if title not in sections:
                sections[title] = [page, page]
            else:
                sections[title][0] = min(sections[title][0], page)
                sections[title][1] = max(sections[title][1], page)

        return [SectionInfo(section_title=t, page_start=b[0], page_end=b[1]) for t, b in sections.items()]

    def compare_documents(self, file_ids: list, query: str, top_k_per_file: int = 5) -> ComparisonResult:
        per_file = {}
        for file_id in file_ids:
            per_file[file_id] = self.search_within_file(file_id, query, top_k=top_k_per_file)
        return ComparisonResult(per_file_findings=per_file)

    def search_for_contradictions(self, claim: str, file_ids: list = None) -> list:
        negated_query = f"evidence that contradicts or is inconsistent with: {claim}"
        return self.search_documents(negated_query, file_ids=file_ids)

    def verify_chunk_supports_claim(self, chunk_id: str, claim: str) -> VerificationResult:
        chunk = self.get_chunk(chunk_id)
        prompt = VERIFY_PROMPT.format(text=chunk.text, claim=claim)

        client = self.llm_provider.get_client()
        response = ask_llm(client, prompt)

        supported = response.strip().upper().startswith("SUPPORTED")
        reasoning = response.split("|", 1)[1].strip() if "|" in response else response.strip()
        return VerificationResult(supported=supported, reasoning=reasoning)
