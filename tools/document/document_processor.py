
import asyncio
import json
import logging
import re
import time

from config import get_settings
from llm_provider import LLMProvider
from tools.document.token_estimate import estimate_tokens
from tools.llm_call import ask_llm_async
from tools.orchestrator.models import DocumentFindings

logger = logging.getLogger("tools.document.processor")

# --- Tunables (configurable via settings, see DocumentProcessor.__init__) --------------------

# Batch by total estimated token count, not chunk count - see module docstring's algorithm and
# the batching example in the design doc this implements (MAX_BATCH_TOKENS = 6000 there).
DEFAULT_MAX_BATCH_TOKENS = 6000

# Cap on concurrent in-flight batch LLM calls - "parallel where possible" doesn't mean
# unbounded; this keeps a very long document from firing 50+ simultaneous requests at the
# provider.
DEFAULT_MAX_PARALLEL_BATCHES = 6


def get_model_config() -> dict:
    """Same per-agent-config pattern as agents/*/config.py - independently tunable from
    DOCUMENT_AGENT_PROVIDER/MODEL since this pipeline makes many small parallel extraction
    calls (cheap/fast matters more here) plus one larger synthesis call, a different cost
    profile than the Document Agent's own reasoning loop. Falls back to LLMProvider's own
    DEFAULT_LLM_PROVIDER when unset."""
    settings = get_settings()
    return {
        "provider": settings.get("DOCUMENT_PROCESSOR_PROVIDER", "") or None,
        "model": settings.get("DOCUMENT_PROCESSOR_MODEL", "") or None,
    }


# --- Prompts -------------------------------------------------------------------------------
# Deliberately intent-agnostic: the SAME two-key JSON shape (summary + findings) is asked for
# no matter what the objective actually says. "summary" and "findings" are generic enough to
# hold a summarization's key points, an anomaly list, extracted risks, action items, FAQ
# entries, whatever - the objective's own wording is what shapes their CONTENT, the KEYS never
# change. That's what lets merge_batch_outputs stay a fixed, deterministic function regardless
# of which free-form objective was given.

_BATCH_SYSTEM_PROMPT = """You are an information extractor, not a chat assistant. You process \
ONE portion of a larger document at a time and return ONLY structured JSON - no commentary, no \
markdown fences, nothing outside the JSON object. Do not answer as if this portion were the \
whole document - only extract what's relevant to the objective below from THIS portion.

Objective (given by the user, applies to the whole document, not just this portion - follow it \
exactly, do not reinterpret or narrow it):
{objective}

Return ONLY a JSON object with EXACTLY these keys:
- "relevant": true or false - whether this portion contains anything relevant to the objective
- "summary": a short string capturing what's relevant to the objective in this portion (empty \
string if nothing relevant)
- "findings": a JSON array of discrete, self-contained points relevant to the objective found \
in this portion - each a complete short statement (a fact, an issue, an item, an answer, \
whatever the objective actually calls for) that would still make sense out of context on its \
own (empty array if nothing relevant)

No extra keys.
"""

_BATCH_USER_PROMPT = """Document: {filename}
Portion {batch_index} of {batch_count} (pages {page_range}):

{batch_text}
"""

_SYNTHESIS_SYSTEM_PROMPT = """You are writing the final answer to a user's request about a \
document. Every portion of the document has already been read and had relevant information \
extracted by an earlier pass - the "merged findings" below are that raw extracted material, \
not a finished answer. You are the ONLY step that writes the user-facing answer.

Merge duplicate findings, rank them by importance where relevant, and produce ONE coherent, \
well-written response that directly satisfies the objective below. Cite specifics from the \
merged findings rather than describing that findings exist. Do not mention "portions", \
"batches", or the extraction process itself - write as if you read the whole document yourself.
"""

_SYNTHESIS_USER_PROMPT = """Objective:
{objective}

Merged findings extracted from the full document:
{merged_json}

Write the final answer now.
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw_text: str) -> dict | None:
    """Same tolerant extraction idiom used elsewhere in this codebase for LLM JSON replies
    (see shared/intent_classifier.py) - pull the first {...} block out rather than assuming
    the whole reply is clean JSON, since instruct models occasionally wrap it anyway."""
    match = _JSON_BLOCK_RE.search(raw_text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# --- Retrieval + batching (pure, no LLM) ----------------------------------------------------

def _ordered_chunks(vector_store, file_ids: list) -> list:
    """Every chunk belonging to the given files, in original document order - grouped by file
    (in the given file order), sorted within each file by chunk_index. No semantic search, no
    query, no ranking: this is a full, deterministic read of the assigned document(s)."""
    ordered = []
    for file_id in file_ids:
        chunks = vector_store.get_by_filter({"file_id": file_id})
        chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
        ordered.extend(chunks)
    return ordered


def batch_chunks(chunks: list, max_batch_tokens: int) -> list:
    """Token-aware batching - see module docstring's algorithm. A single chunk larger than
    max_batch_tokens on its own still gets its own batch rather than being dropped or split."""
    batches: list = []
    current: list = []
    current_tokens = 0
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk.text)
        if current and current_tokens + chunk_tokens > max_batch_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(chunk)
        current_tokens += chunk_tokens
    if current:
        batches.append(current)
    return batches


def merge_batch_outputs(batch_outputs: list) -> dict:
    """Combine every batch's JSON into one merged object - no LLM involved, and no knowledge of
    what the objective actually asked for (the keys are always exactly "summary"/"findings", see
    _BATCH_SYSTEM_PROMPT). Non-relevant/empty batches contribute nothing. "findings" items are
    deduped (dict/list items on their JSON form since they aren't hashable, everything else on
    its normalized string form) - "summary" strings are simply joined."""
    summaries = []
    findings: list = []
    seen = set()
    for b in batch_outputs:
        if not isinstance(b, dict):
            continue
        summary = str(b.get("summary", "")).strip()
        if summary:
            summaries.append(summary)
        for item in (b.get("findings") or []):
            marker = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else str(item).strip().lower()
            if not marker or marker in seen:
                continue
            seen.add(marker)
            findings.append(item)

    return {"summary": "\n\n".join(summaries), "findings": findings}


class DocumentProcessor:
    """One-shot pipeline instance - construct fresh per invoke_document_processor call, same
    lifecycle as TabularAgent/DocumentAgent (see tools/orchestrator/orchestrator_tools.py)."""

    def __init__(self, assigned_files: list, vector_store, llm_provider=None, max_batch_tokens: int = None):
        self.assigned_file_ids = [f.file_id for f in assigned_files]
        # FileRef (tools/orchestrator/models.py) only carries file_id - no filename - so this
        # falls back to the id itself; it's only used as informational text in batch prompts,
        # never for lookup.
        self.filenames = {fid: fid for fid in self.assigned_file_ids}
        self.vector_store = vector_store
        if llm_provider is not None:
            self.llm_provider = llm_provider
            self._model = None
        else:
            model_config = get_model_config()
            self.llm_provider = LLMProvider(model_config["provider"])
            self._model = model_config["model"]

        settings = get_settings()
        self.max_batch_tokens = int(
            max_batch_tokens
            or settings.get("DOCUMENT_PROCESSOR_MAX_BATCH_TOKENS", DEFAULT_MAX_BATCH_TOKENS)
            or DEFAULT_MAX_BATCH_TOKENS
        )
        self.max_parallel_batches = int(
            settings.get("DOCUMENT_PROCESSOR_MAX_PARALLEL_BATCHES", DEFAULT_MAX_PARALLEL_BATCHES)
            or DEFAULT_MAX_PARALLEL_BATCHES
        )
        self.logger = logger

    async def run(self, objective: str, constraints: dict = None, on_event=None) -> DocumentFindings:
        """`objective` is a free-form analysis instruction the orchestrator wrote from the
        user's request (e.g. "Identify inconsistencies in this financial report.") - used
        VERBATIM as the reasoning instruction for every batch and the final synthesis call. See
        module docstring for why this is deliberately not a fixed intent enum."""
        run_start = time.perf_counter()

        if on_event is not None:
            await on_event({"type": "tool_call", "message": "Reading the full document"})

        chunks = await asyncio.to_thread(_ordered_chunks, self.vector_store, self.assigned_file_ids)
        if not chunks:
            self.logger.warning(
                "document_processor: no chunks found for file_ids=%s", self.assigned_file_ids,
            )
            return DocumentFindings(
                summary="No content was found for the assigned file(s), so no analysis could be produced.",
                artifact_refs=[], source_refs=[],
            )

        batches = batch_chunks(chunks, self.max_batch_tokens)
        self.logger.info(
            "document_processor: %d chunk(s) across %d file(s) -> %d batch(es) "
            "(max_batch_tokens=%d, max_parallel=%d, objective=%r)",
            len(chunks), len(self.assigned_file_ids), len(batches),
            self.max_batch_tokens, self.max_parallel_batches, objective,
        )

        semaphore = asyncio.Semaphore(self.max_parallel_batches)

        async def _bounded(batch, idx):
            async with semaphore:
                return await self._process_batch(batch, idx, len(batches), objective)

        batch_outputs = await asyncio.gather(
            *(_bounded(batch, idx) for idx, batch in enumerate(batches, start=1))
        )

        merged = merge_batch_outputs(batch_outputs)

        if on_event is not None:
            await on_event({"type": "tool_call", "message": "Synthesizing the final answer"})

        final_text = await self._synthesize(objective, merged)

        source_refs = list(dict.fromkeys(c.chunk_id for c in chunks))
        self.logger.info(
            "document_processor run took %.3fs (%d batches)",
            time.perf_counter() - run_start, len(batches),
        )
        return DocumentFindings(summary=final_text, artifact_refs=[], source_refs=source_refs)

    async def _process_batch(self, batch: list, batch_index: int, batch_count: int, objective: str) -> dict:
        pages = sorted({c.metadata.get("page", 0) for c in batch})
        page_range = f"{pages[0]}-{pages[-1]}" if pages else "unknown"
        filename = self.filenames.get(batch[0].file_id, batch[0].file_id)
        batch_text = "\n\n".join(c.text for c in batch)

        prompt = (
            _BATCH_SYSTEM_PROMPT.format(objective=objective)
            + "\n\n"
            + _BATCH_USER_PROMPT.format(
                filename=filename, batch_index=batch_index, batch_count=batch_count,
                page_range=page_range, batch_text=batch_text,
            )
        )

        client = self.llm_provider.get_client(self._model)
        try:
            raw = await ask_llm_async(client, prompt)
        except Exception:
            self.logger.exception("document_processor: batch %d/%d failed", batch_index, batch_count)
            return {}

        parsed = _extract_json(raw)
        if parsed is None:
            self.logger.warning(
                "document_processor: batch %d/%d returned unparseable output, treating as empty: %r",
                batch_index, batch_count, raw[:200] if raw else raw,
            )
            return {}
        return parsed

    async def _synthesize(self, objective: str, merged: dict) -> str:
        client = self.llm_provider.get_client(self._model)
        prompt = (
            _SYNTHESIS_SYSTEM_PROMPT
            + "\n\n"
            + _SYNTHESIS_USER_PROMPT.format(objective=objective, merged_json=json.dumps(merged, indent=2))
        )
        try:
            result = await ask_llm_async(client, prompt)
            return result.strip()
        except Exception:
            self.logger.exception("document_processor: final synthesis call failed")
            # Best-effort fallback so a synthesis-call failure doesn't lose everything the
            # batches already extracted.
            return (
                "Could not synthesize a final answer due to an internal error. Raw extracted "
                f"findings:\n{json.dumps(merged, indent=2)}"
            )
