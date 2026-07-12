from llm_provider import LLMProvider
from tools.document.models import (
    ChunkResult,
    ComparisonResult,
    FileOverview,
    SectionInfo,
    TableInfo,
    VerificationResult,
)
from tools.llm_call import ask_llm

VERIFY_PROMPT = """Chunk text:
{text}

Claim: {claim}

Does the chunk text support the claim? Answer with SUPPORTED or NOT_SUPPORTED, then a pipe |, then a one sentence reason.
"""

BROAD_SCAN_PROMPT = """You are scanning one section of a longer document to help with this request:
{focus}

Read this section and write down whatever is relevant to that request - a summary point, a
possible problem, an anomaly, a key fact - whatever the request actually needs. If nothing in
this section is relevant, say so in one short line. Be concise.

Document section (pages {page_range}):
{batch_text}
"""

EXPAND_QUERY_PROMPT = """A user asked this question about a document:
{objective}

Write 3 alternate ways to phrase a search query for finding relevant passages - using
different wording, synonyms, or related terms than the original, since relevant text doesn't
always use the same words as the question. Reply with exactly 3 lines, one reformulation per
line, nothing else.
"""


class DocumentTools:
    def __init__(self, assigned_files: list, vector_store, reranker=None, llm_provider=None):
        self.assigned_file_ids = [f.file_id for f in assigned_files]
        self.vector_store = vector_store
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

    def _to_table_info(self, chunk) -> TableInfo:
        meta = chunk.metadata
        columns = [c.strip() for c in meta.get("columns", "").split(",") if c.strip()]
        caption = chunk.text.split("\nColumns:")[0].strip()
        return TableInfo(
            table_ref=meta.get("table_ref", ""),
            page=meta.get("page", 0),
            caption=caption,
            columns=columns,
            row_count=meta.get("row_count", 0),
        )

    def get_file_overview(self, file_id: str) -> FileOverview:
        """Orient yourself on a file before doing anything else: its sections, its tables, and
        a quick list of key topics (from section headings). Call this first for most
        objectives on a file you haven't looked at yet, the same way you'd skim a table of
        contents before reading a report."""
        sections = self.list_file_sections(file_id)
        tables = self.list_tables(file_id)
        key_topics = [s.section_title for s in sections if s.section_title][:10]
        return FileOverview(file_id=file_id, sections=sections, tables=tables, key_topics=key_topics)

    def expand_query(self, objective: str) -> list:
        """Generate 3 alternate phrasings of the objective, to widen semantic search recall
        beyond its literal wording. Call this when an initial search_documents call comes up
        thin or off-target, then re-search using the original objective plus a couple of these
        reformulations - relevant text doesn't always match the question's exact words."""
        client = self.llm_provider.get_client()
        response = ask_llm(client, EXPAND_QUERY_PROMPT.format(objective=objective))
        return [line.strip("- ").strip() for line in response.strip().splitlines() if line.strip()]

    def search_documents(self, query: str, file_ids: list = None, top_k: int = 8) -> list:
        """Semantic search across assigned files (or a subset via file_ids) for chunks relevant
        to query. Call this first for most objectives - it's the main way to find relevant text.
        Returns up to top_k ChunkResults (chunk_id, file_id, text, score, metadata)."""
        scoped = self._scoped_file_ids(file_ids)
        filters = {"file_id": {"$in": scoped}}
        fetch_k = top_k * 3 if self.reranker else top_k

        chunks = self.vector_store.query(query, fetch_k, filters=filters)
        if self.reranker:
            chunks = self.reranker.rank(query, chunks, top_k=top_k)

        return [self._to_result(c) for c in chunks[:top_k]]

    def search_within_file(self, file_id: str, query: str, top_k: int = 8) -> list:
        """Semantic search restricted to one file. Use this instead of search_documents when the
        objective already tells you which document to look in."""
        return self.search_documents(query, file_ids=[file_id], top_k=top_k)

    def get_chunk(self, chunk_id: str) -> ChunkResult:
        """Fetch one chunk by its exact chunk_id. Use this to re-read a chunk you already found
        (e.g. from a search result) instead of searching again."""
        results = self.vector_store.get_by_id([chunk_id])
        if not results:
            raise ValueError(f"chunk_id '{chunk_id}' not found")
        return self._to_result(results[0])

    def get_surrounding_chunks(self, chunk_id: str, window: int = 1) -> list:
        """Fetch the chunks immediately before/after a chunk_id (up to window chunks on each
        side, same file, ordered by position). Use this when a chunk's text seems cut off and
        you need more surrounding context to understand it fully."""
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
        """List every section/heading in a file with its page range. Use this to understand a
        long document's structure before deciding where to search."""
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
        """Run the same query separately against each of the given files and return per-file
        results. Use this when the objective asks you to compare, contrast, or find differences
        across multiple documents."""
        per_file = {}
        for file_id in file_ids:
            per_file[file_id] = self.search_within_file(file_id, query, top_k=top_k_per_file)
        return ComparisonResult(per_file_findings=per_file)

    def search_for_contradictions(self, claim: str, file_ids: list = None) -> list:
        """Search for chunks that might contradict or be inconsistent with a specific claim.
        Use this to stress-test a finding before reporting it as confident."""
        negated_query = f"evidence that contradicts or is inconsistent with: {claim}"
        return self.search_documents(negated_query, file_ids=file_ids)

    def list_tables(self, file_id: str) -> list:
        """List every table extracted from a file - page, caption, columns, row_count, and
        table_ref (the id a Tabular Agent can query for the table's actual values). No search
        ranking involved, this returns every table for the file deterministically. Use this
        (or search_tables) instead of guessing via search_documents whenever the objective is
        about what tables exist in a document."""
        filters = {"$and": [{"file_id": file_id}, {"type": "table"}]}
        chunks = self.vector_store.get_by_filter(filters)
        return [self._to_table_info(c) for c in chunks]

    def search_tables(self, query: str, file_ids: list = None, top_k: int = 5) -> list:
        """Semantic search restricted to tables only (their captions and column names), so
        prose chunks don't crowd out table matches. Use this instead of search_documents when
        the objective is about finding a specific table by topic, e.g. "the table showing
        annual returns"."""
        scoped = self._scoped_file_ids(file_ids)
        filters = {"$and": [{"file_id": {"$in": scoped}}, {"type": "table"}]}
        chunks = self.vector_store.query(query, top_k, filters=filters)
        return [self._to_table_info(c) for c in chunks]

    def get_table(self, table_ref: str) -> TableInfo:
        """Fetch one table's info (page, caption, columns, row_count) by its table_ref. This
        does NOT return the table's actual row data - that lives in a Parquet file. If the
        objective needs the real values or any computation over a table, report its table_ref
        so the Tabular Agent can be invoked on it - never fabricate or compute table values
        yourself."""
        chunks = self.vector_store.get_by_filter({"table_ref": table_ref})
        if not chunks:
            raise ValueError(f"table_ref '{table_ref}' not found")
        return self._to_table_info(chunks[0])

    def broad_scan(self, file_id: str, focus: str, batch_size: int = 8) -> str:
        """Read an entire file section by section (not just a similarity-ranked slice) and
        collect what's relevant to focus from every section, in order. Use this instead of
        search_documents when the objective needs the WHOLE document considered - e.g.
        "summarize this file", "is anything wrong in this file", "find any anomalies" - cases
        where top-k semantic search could easily miss parts of the document. Slower and costs
        more tokens than search_documents, so only reach for it when full-document coverage is
        actually needed. Returns each section's findings concatenated, tagged with page ranges,
        for you to synthesize into a final answer."""
        chunks = self.vector_store.get_by_filter({"file_id": file_id})
        chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))

        client = self.llm_provider.get_client()
        results = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            batch_text = "\n\n".join(c.text for c in batch)
            pages = sorted({c.metadata.get("page", 0) for c in batch})
            page_range = f"{pages[0]}-{pages[-1]}" if pages else "unknown"

            prompt = BROAD_SCAN_PROMPT.format(focus=focus, page_range=page_range, batch_text=batch_text)
            output = ask_llm(client, prompt)
            results.append(f"[pages {page_range}]\n{output}")

        return "\n\n".join(results)

    def verify_chunk_supports_claim(self, chunk_id: str, claim: str) -> VerificationResult:
        """Check whether a specific chunk's text actually supports a claim (returns
        supported: bool, reasoning: str). Use this as a final sanity check on a chunk you plan
        to cite, especially if you're not fully sure it says what you think it says."""
        chunk = self.get_chunk(chunk_id)
        prompt = VERIFY_PROMPT.format(text=chunk.text, claim=claim)

        client = self.llm_provider.get_client()
        response = ask_llm(client, prompt)

        supported = response.strip().upper().startswith("SUPPORTED")
        reasoning = response.split("|", 1)[1].strip() if "|" in response else response.strip()
        return VerificationResult(supported=supported, reasoning=reasoning)
