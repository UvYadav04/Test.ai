# Agentic Data Analysis Workspace — Complete Tool Specification

## 0. Purpose of This Document

This document specifies every tool available to every agent/component in the system:

- **Main Orchestrator** — file catalog & discovery tools, plus the delegation calls it uses to spawn subagents
- **Tabular Agent** — CSV/tabular query tools (backed by DuckDB)
- **Document Agent** — RAG/document investigation tools (backed by the vector store)
- **Hypothesis Generation** — specified here as a tool-shaped bounded LLM call, even though it is not an agent

Each tool includes: purpose, signature, return shape, and implementation notes for Claude Code. This is a specification document, not runnable code — use it to generate the actual tool implementations.

---

# 1. Main Orchestrator Tools

The orchestrator's tools operate only on **shallow metadata** captured at ingestion time (filenames, types, schema summaries, row/page counts). The orchestrator must never open a file's actual content directly — that is exclusively the job of the subagents it delegates to.

## 1.1 `list_files`

**Purpose:** The base file discovery tool. Returns files in a workspace matching a structured filter. This is the primary tool the orchestrator uses to figure out what data exists before deciding what to delegate.

```python
def list_files(
    workspace_id: str,
    filters: dict = None,
    max_results: int = 20,
) -> list[FileCatalogEntry]
```

**Filter shape:**
```python
filters = {
    "name_contains": str,             # substring match on filename, case-insensitive
    "file_type": list[str],           # e.g. ["csv", "pdf"]
    "uploaded_after": str,            # ISO date
    "uploaded_before": str,           # ISO date
    "min_rows": int,                  # tabular files only, ignored for pdf
    "max_rows": int,
    "tags": list[str],                # if user-applied tags/categories are supported
}
```

**Return shape:**
```python
FileCatalogEntry:
  file_id: str
  filename: str
  file_type: Literal["csv", "json", "pdf"]
  uploaded_at: datetime
  size_bytes: int
  row_count: Optional[int]       # tabular only
  page_count: Optional[int]      # pdf only
  columns: Optional[list[str]]   # tabular only, from schema_summary
  tags: Optional[list[str]]
```

**Design note:** One tool with a structured filter dict, not several narrow tools (`list_files_by_name`, `list_files_by_date`, etc.). Real orchestrator queries are compound ("CSV files uploaded in Q3 with 'revenue' in the name" = 3 conditions in one call), and a single filterable tool expresses that in one call rather than forcing the orchestrator to over-fetch and intersect manually.

**Implementation notes for Claude Code:**
- Backed by a workspace file metadata table/store (populated at ingestion time from each `IngestionResult.schema_summary`) — not a live filesystem scan.
- `max_results` caps the payload by default; if a workspace has hundreds of files, force the orchestrator to narrow filters rather than silently returning everything.
- Filtering should be pushed down to the metadata store query (e.g., SQL WHERE clauses or equivalent), not done in Python after fetching everything.

## 1.2 `search_files`

**Purpose:** Fuzzy/semantic search over filenames and any extracted titles/section headers, for when the user's phrasing doesn't literally match a filename (e.g., user says "the churn numbers," file is `customer_retention_q3.csv`).

```python
def search_files(
    workspace_id: str,
    query: str,
    max_results: int = 10,
) -> list[FileCatalogEntry]
```

**Implementation notes for Claude Code:**
- Start with a simple fuzzy string match (e.g., `rapidfuzz` library) over filenames + any `schema_summary` section titles for PDFs.
- Do not build embedding-based filename search in the initial version — not worth the complexity until fuzzy match proves insufficient in practice.

## 1.3 `get_file_details`

**Purpose:** Fetch full catalog metadata for one already-known file_id, without re-running a full list query. Used when the orchestrator already has a `file_id` from a prior turn's `selected_files` in Investigation State and just needs a fresh check.

```python
def get_file_details(file_id: str) -> FileCatalogEntry
```

**Implementation notes for Claude Code:** Simple lookup by primary key against the same metadata store `list_files` queries. Should raise a clear error if `file_id` doesn't exist (e.g., file was deleted).

## 1.4 Delegation Tools

**Purpose:** The calls the orchestrator uses to spawn subagents. These are the boundary across which context isolation is enforced — only `objective`, `assigned_files`, and `constraints` cross into the subagent; only the structured findings object crosses back.

```python
def invoke_tabular_agent(
    objective: str,
    assigned_files: list[FileRef],
    constraints: dict = None,
) -> TabularFindings

def invoke_document_agent(
    objective: str,
    assigned_files: list[FileRef],
    constraints: dict = None,
) -> DocumentFindings
```

```python
FileRef:
  file_id: str
  output_ref: str    # parquet path (tabular) or vectordb collection ref (document)
```

**Critical implementation rule:** `FileRef` is deliberately minimal. Do NOT pass the full `FileCatalogEntry` (with columns, row counts, etc.) into the subagent call as a shortcut. The subagent must independently call its own `inspect_schema()`/`extract_metadata()` tools to learn about the file. Passing the orchestrator's cached catalog metadata down would let a stale or shallow summary silently substitute for the subagent's own verification — this breaks the context isolation principle and can cause the subagent to trust wrong assumptions about a file's structure.

**`constraints` shape (freeform but conventionally includes):**
```python
constraints = {
    "date_range": {"start": "...", "end": "..."},
    "filters": {...},           # any pre-known filter conditions from the user's question
    "prior_findings_summary": str,  # optional, only if this is a follow-up delegation in the same investigation round
}
```

---

# 2. Tabular Agent Tools

Backed by DuckDB, operating on Parquet files produced by the ingestion pipeline. The agent runs its own internal loop across these tools before returning one final structured result — none of the intermediate tool calls or raw query outputs are visible to the orchestrator.

## 2.1 `list_allowed_files`

**Purpose:** Returns metadata only for the files assigned to this specific invocation — the agent's hard-scoped view. It should be structurally impossible for this tool to return a file the orchestrator didn't assign.

```python
def list_allowed_files() -> list[FileMetadata]

FileMetadata:
  file_id: str
  filename: str
  output_ref: str
  row_count: int
  columns: list[str]
```

**Implementation notes for Claude Code:** Scope this tool to the `assigned_files` passed into the current agent invocation — implement as a closure/instance attribute set at agent construction time, not a parameter the LLM can freely choose.

## 2.2 `inspect_schema`

**Purpose:** Deep schema inspection of one assigned file — the agent's own verification, independent of any orchestrator-level cache.

```python
def inspect_schema(file_id: str) -> SchemaInfo

SchemaInfo:
  columns: list[str]
  dtypes: dict[str, str]
  nullable: dict[str, bool]
  sample_size: int
  likely_key_columns: list[str]   # heuristic: columns named *_id, or high-cardinality unique columns
```

**Implementation notes for Claude Code:** Read Parquet schema via DuckDB (`DESCRIBE SELECT * FROM read_parquet(...)`) or pyarrow's schema reader — don't load the full file into memory just to inspect schema. `likely_key_columns` can be a simple heuristic (column name ends in `_id`, or `COUNT(DISTINCT col) ≈ COUNT(*)`).

## 2.3 `sample_rows`

**Purpose:** Preview actual data values/format before writing a query — catches format assumptions that would otherwise cause silent bugs (e.g., `revenue` stored as `"$1,200.00"` string vs. float, `date` stored as `"Q3-2026"` text vs. real date type).

```python
def sample_rows(file_id: str, n: int = 10) -> list[dict]
```

**Implementation notes for Claude Code:** `SELECT * FROM read_parquet(...) LIMIT n` via DuckDB. Keep `n` capped at a reasonable max (e.g., 50) regardless of what's requested, to avoid the agent accidentally pulling large samples into its own context repeatedly.

## 2.4 `find_join_candidates`

**Purpose:** Given multiple assigned files, suggests likely join keys by checking column name overlap plus sampled value overlap — needed for cross-file questions (e.g., "which customers churned and what was their total revenue").

```python
def find_join_candidates(file_ids: list[str]) -> list[JoinCandidate]

JoinCandidate:
  file_a: str
  column_a: str
  file_b: str
  column_b: str
  match_confidence: float   # 0-1, based on name similarity + sampled value overlap
```

**Implementation notes for Claude Code:** Start simple — compare column names for exact/fuzzy matches across files, then for candidate pairs, sample values from both columns and compute overlap ratio (e.g., `len(set_a & set_b) / len(set_a)`). No need for a sophisticated schema-matching library initially.

## 2.5 `query_data`

**Purpose:** The core execution tool — runs SQL directly against the registered Parquet files. This is the workhorse; most of the agent's iteration happens here.

```python
def query_data(
    sql: str,
    file_ids: list[str],
    row_cap: int = 500,
    timeout_seconds: int = 15,
) -> QueryResult

QueryResult:
  columns: list[str]
  rows: list[dict]
  row_count: int
  truncated: bool
  error: Optional[str]
```

**Design decision (per prior discussion):** Allow raw SQL rather than building a constrained query DSL. DuckDB is fast and local (not hitting shared production infrastructure), so the risk profile is low, and a raw-SQL failure (a DuckDB error) is easy for the agent to catch and retry against — building and maintaining a constrained DSL is not worth the engineering effort here.

**Implementation notes for Claude Code:**
- Register each `file_id` as a DuckDB view: `CREATE VIEW {file_id} AS SELECT * FROM read_parquet('{output_ref}')`, scoped to only files in `file_ids` for this call — do not expose views for files outside the current agent's assignment.
- Wrap execution with a timeout (DuckDB supports query cancellation via a separate thread + timer, or use `duckdb.execute` with a watchdog).
- Enforce `row_cap` via `LIMIT` injection or post-query truncation; set `truncated=True` and include the true `row_count` (via a `COUNT(*)` wrapper or `SELECT COUNT(*) FROM (...)`) so the agent knows if it needs to aggregate further rather than assume it saw everything.
- On SQL error, return `error` populated and let the agent see it and retry — do not raise an exception up through the tool boundary.

## 2.6 `aggregate`

**Purpose:** A structured convenience wrapper over `query_data` for the common case (group-by + sum/avg/count/min/max), so the agent doesn't have to hand-write SQL for every simple aggregation — reduces SQL-generation error rate on the most frequent operation type.

```python
def aggregate(
    file_ids: list[str],
    group_by: list[str],
    metrics: list[MetricSpec],
    filters: Optional[dict] = None,
) -> QueryResult

MetricSpec:
  column: str
  op: Literal["sum", "avg", "count", "min", "max"]
  alias: Optional[str]
```

**Implementation notes for Claude Code:** This can be implemented as a thin builder that generates the equivalent SQL and calls the same underlying execution path as `query_data` (same row cap, timeout, and error handling) — don't duplicate the execution logic.

## 2.7 `describe_column`

**Purpose:** Cheap profiling of a single column before trusting it in an aggregation — catches financial data quirks (negative revenue from refunds mixed in, test/demo rows, currency inconsistencies) before they silently corrupt a sum.

```python
def describe_column(file_id: str, column: str) -> ColumnProfile

ColumnProfile:
  min: Any
  max: Any
  mean: Optional[float]      # numeric columns only
  null_count: int
  distinct_count: int
  top_values: list[tuple]    # [(value, count), ...] top 5-10
```

**Implementation notes for Claude Code:** A single DuckDB aggregate query per call (`SELECT MIN(col), MAX(col), AVG(col), COUNT(*) FILTER (WHERE col IS NULL), COUNT(DISTINCT col) FROM ...`), plus a separate `GROUP BY col ORDER BY COUNT(*) DESC LIMIT 10` for `top_values`.

## 2.8 `validate_result`

**Purpose:** Self-check tool the agent calls before finalizing a finding — sanity-checks row counts, null-heavy columns, and obvious outliers, and supports re-computing the same aggregation a different way for cross-verification.

```python
def validate_result(
    result: QueryResult,
    expected_shape: Optional[dict] = None,
) -> ValidationReport

ValidationReport:
  passed: bool
  warnings: list[str]
```

**Implementation notes for Claude Code:** Rule-based checks to start — e.g., flag if `row_count == 0` when the query implied data should exist, flag if a "sum" metric came back negative when the column is a revenue-type field, flag if `null_count` on a used column exceeds some threshold (e.g., 20% of rows). This does not need to be an LLM call — keep it deterministic per the project's core design principle.

---

# 3. Document Agent Tools

Backed by the vector store, operating on chunks produced by the PDF ingestion pipeline. Raw chunk content stays inside this agent's own context — never passed up to the orchestrator; only compact findings with `chunk_refs`/`artifact_refs` cross the boundary.

## 3.1 `search_documents`

**Purpose:** The primary retrieval tool — semantic search over the vector store, scoped to the agent's assigned files by default.

```python
def search_documents(
    query: str,
    file_ids: Optional[list[str]] = None,   # defaults to all assigned files
    top_k: int = 8,
) -> list[ChunkResult]

ChunkResult:
  chunk_id: str
  file_id: str
  text: str
  score: float
  metadata: dict   # { page, section, chunk_index, source_file }
```

**Implementation notes for Claude Code:** Embed `query` using the same embedding model used at ingestion time, call `vector_store.query(embedding, top_k, filters={"file_id": {"$in": file_ids}})` if `file_ids` given, else default filter to the full `assigned_files` list for this invocation — never allow an unfiltered query across the whole workspace's vector store.

## 3.2 `search_within_file`

**Purpose:** Narrower version of `search_documents` for when the agent already knows which single file is relevant — avoids noise from other assigned files diluting results.

```python
def search_within_file(
    file_id: str,
    query: str,
    top_k: int = 8,
) -> list[ChunkResult]
```

**Implementation notes for Claude Code:** Same as `search_documents` but with a hard single-file filter — can be implemented as a thin wrapper calling `search_documents(query, file_ids=[file_id], top_k=top_k)`.

## 3.3 `get_chunk`

**Purpose:** Fetch one specific chunk by ID directly — for re-examining a chunk found earlier without re-running a full search.

```python
def get_chunk(chunk_id: str) -> ChunkResult
```

**Implementation notes for Claude Code:** Direct call to `vector_store.get_by_id([chunk_id])`, return the single result.

## 3.4 `get_surrounding_chunks`

**Purpose:** Returns neighboring chunks (by `chunk_index` within the same file) around a given chunk — a single retrieved chunk is often missing context (a number/clause only makes sense with the sentence before/after it). Prevents citing facts out of context.

```python
def get_surrounding_chunks(
    chunk_id: str,
    window: int = 1,
) -> list[ChunkResult]
```

**Implementation notes for Claude Code:** Look up the target chunk's `file_id` and `chunk_index` from its metadata, then query the vector store's metadata filter for the same `file_id` with `chunk_index` in `[index - window, index + window]`, ordered by `chunk_index`. If the vector store's native filtering can't easily do a range query on `chunk_index`, fall back to fetching by constructing the expected chunk_ids directly (since `chunk_id = f"{file_id}_{chunk_index}"` per the ingestion spec) and calling `get_by_id`.

## 3.5 `list_file_sections`

**Purpose:** Lightweight table-of-contents view of a document's structure, from metadata captured at ingestion (section headers, page boundaries). Lets the agent orient itself in a long document before searching blindly.

```python
def list_file_sections(file_id: str) -> list[SectionInfo]

SectionInfo:
  section_title: str
  page_start: int
  page_end: int
```

**Implementation notes for Claude Code:** Depends on section metadata being captured at ingestion time (from Docling's structure detection — see PDF ingestion phase). If section titles aren't reliably detected for a given PDF, return an empty list or page-range-only entries rather than failing.

## 3.6 `compare_documents`

**Purpose:** Convenience wrapper that runs the same query across multiple files individually and returns results grouped by file, so the agent can spot contradictions/differences without manually issuing and comparing N separate searches.

```python
def compare_documents(
    file_ids: list[str],
    query: str,
    top_k_per_file: int = 5,
) -> ComparisonResult

ComparisonResult:
  per_file_findings: dict[str, list[ChunkResult]]   # keyed by file_id
```

**Implementation notes for Claude Code:** Implement as a loop calling `search_within_file` for each `file_id`, collecting results into the per-file dict — no new retrieval logic needed, just an orchestration convenience.

## 3.7 `search_for_contradictions`

**Purpose:** Deliberately searches for evidence that conflicts with a stated claim, rather than only confirming it — prevents the common RAG failure mode of confirmation-biased retrieval.

```python
def search_for_contradictions(
    claim: str,
    file_ids: Optional[list[str]] = None,
) -> list[ChunkResult]
```

**Implementation notes for Claude Code:** Implement via a contrastive/negated query framing — e.g., prefix the claim with a negation template ("evidence that contradicts or is inconsistent with: {claim}") before embedding and searching, or issue two searches (one for the claim as-is, one for a negated version) and return results that don't overlap with the confirming search. Start with the simpler negated-query-framing approach; it doesn't require any new infrastructure beyond `search_documents`.

## 3.8 `verify_chunk_supports_claim`

**Purpose:** A lightweight, bounded verification call (not a full loop) the Document Agent runs before finalizing a finding — checks whether a specific chunk actually supports a specific claim. This is included in the initial build (not deferred) because it's cheap and directly prevents the most common RAG failure mode: citing a chunk that doesn't actually say what the agent claims it says.

```python
def verify_chunk_supports_claim(
    chunk_id: str,
    claim: str,
) -> VerificationResult

VerificationResult:
  supported: bool
  reasoning: str
```

**Implementation notes for Claude Code:** Implement as a single bounded LLM call (no tools, no loop) — pass the chunk's text and the claim, ask the model to judge support with a short reasoning string. This is analogous to `validate_result` on the Tabular Agent side, but since it requires semantic judgment (not just numeric sanity-checking), it needs an LLM call rather than a pure rule-based check.

---

# 4. Hypothesis Generation (Tool-Shaped Bounded LLM Call)

**Purpose:** For complex or "why"-style queries, generates and prioritizes candidate explanations *before* the orchestrator delegates to subagents — so investigation effort focuses on the most likely directions first rather than exploring blindly. This is **not an agent** — no tool use, no loop, a single reasoning call — but it is specified here in tool form so it can be invoked from the orchestrator's tool-calling interface consistently with everything else.

```python
def generate_hypotheses(
    objective: str,
    context: dict,
    max_hypotheses: int = 5,
) -> HypothesisResult

context = {
    "available_files": list[FileCatalogEntry],   # shallow catalog, from list_files/search_files
    "known_findings": Optional[list[dict]],       # if this is a follow-up round in an ongoing investigation
}

HypothesisResult:
  hypotheses: list[Hypothesis]

Hypothesis:
  statement: str              # e.g., "Churn spike is concentrated in the Enterprise tier"
  suggested_investigation: str  # e.g., "check churn.csv broken down by customer tier"
  suggested_agent: Literal["tabular", "document", "both"]
  priority: int                # 1 = highest priority
```

**When the orchestrator should call this:**
```
Simple, direct query        → skip this tool, delegate directly
Complex / "why" query       → call generate_hypotheses() first
                             → delegate to agents in priority order based on results
```

**Implementation notes for Claude Code:**
- Implement as a single LLM call (system prompt instructing it to output structured JSON matching `HypothesisResult`), not an agent with its own tools.
- Input `context.available_files` should come from a prior `list_files`/`search_files` call the orchestrator already made — do not have this tool independently re-query the file catalog; keep it a pure reasoning step over data the orchestrator already gathered.
- `priority` ordering is what the orchestrator should use to decide delegation order — investigate highest-priority hypotheses first, and use the "enough evidence?" check (§4.9 in the main architecture doc) to decide whether lower-priority hypotheses need investigating at all.
- Cache/log the raw hypotheses list into Investigation State's `open_questions` field, so lower-priority hypotheses that weren't investigated are still visible as known gaps rather than silently dropped.

---

# 5. Summary Table — All Tools by Owner

| Owner | Tool | Type |
|---|---|---|
| Orchestrator | `list_files` | deterministic |
| Orchestrator | `search_files` | deterministic (fuzzy match) |
| Orchestrator | `get_file_details` | deterministic |
| Orchestrator | `invoke_tabular_agent` | delegation (spawns agent) |
| Orchestrator | `invoke_document_agent` | delegation (spawns agent) |
| Orchestrator | `generate_hypotheses` | bounded LLM call |
| Tabular Agent | `list_allowed_files` | deterministic |
| Tabular Agent | `inspect_schema` | deterministic |
| Tabular Agent | `sample_rows` | deterministic |
| Tabular Agent | `find_join_candidates` | deterministic |
| Tabular Agent | `query_data` | deterministic (DuckDB) |
| Tabular Agent | `aggregate` | deterministic (DuckDB) |
| Tabular Agent | `describe_column` | deterministic (DuckDB) |
| Tabular Agent | `validate_result` | deterministic (rule-based) |
| Document Agent | `search_documents` | deterministic (vector search) |
| Document Agent | `search_within_file` | deterministic (vector search) |
| Document Agent | `get_chunk` | deterministic |
| Document Agent | `get_surrounding_chunks` | deterministic |
| Document Agent | `list_file_sections` | deterministic |
| Document Agent | `compare_documents` | deterministic (composite) |
| Document Agent | `search_for_contradictions` | deterministic (vector search) |
| Document Agent | `verify_chunk_supports_claim` | bounded LLM call |

This table reflects the project's core design principle throughout: the large majority of tools are deterministic functions; only `generate_hypotheses` and `verify_chunk_supports_claim` are bounded LLM calls; only the two `invoke_*_agent` calls spawn genuine adaptive agent loops.
