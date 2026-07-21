"""LlamaParse-backed PDF parsing - replaces the old docling-native convert_document() that
used to live in utils.py, because docling's own CPU layout/table-structure pipeline measured
858.79s for one dense SEC filing (see worker_service/worker.py's job_timeout comment and
pdf_ingestor.py's git history). LlamaParse (https://www.llamaindex.ai/llamaparse) runs that
same class of work as a hosted service, so ingestion no longer burns worker CPU or risks the
arq job_timeout on conversion time - it just uploads the PDF and waits on a result.

We still chunk with docling's HybridChunker (chunker.py's DoclingChunker) rather than
reinventing heading-aware, token-budgeted chunking - LlamaParse's job here is only to
replace the *parsing* step, not the chunker. To hand LlamaParse's output to HybridChunker
(which requires a real docling DoclingDocument) each LlamaParse page's markdown is run
through docling's OWN Markdown backend (InputFormat.MD -> SimplePipeline). That backend is a
plain GFM/markdown parser with no layout-detection or table-structure models involved, so
routing through it stays cheap - this is not the expensive path docling used to run.

Chunking/table-extraction happens per LlamaParse page (see pdf_ingestor.py's ingest() loop)
rather than concatenating every page's markdown into one blob first, specifically to keep
page numbers accurate: markdown has no native concept of "page", so a single DoclingDocument
built from a multi-page blob would not carry real per-item page provenance. LlamaParse
already reports the true page number for each page, so PDFIngestor uses that directly
instead of trusting docling's (nonexistent, for markdown input) page model.

Uses the v1 `llama-cloud-services` SDK (LlamaParse.aparse()) rather than the newer
`llama-cloud` v2 SDK LlamaIndex started recommending in Jan 2026 - v2's Parse API needs a
two-step upload-then-poll flow through structured config objects, and v1 remains fully
supported (LlamaIndex committed to maintaining llama-cloud-services; only the GitHub repo
gets archived, not the PyPI package). If the project wants v2's newer features later
(bounding boxes, richer image extraction), migrate _get_parser()/parse_pdf_pages() together
per LlamaIndex's migration guide - nothing else in the ingestion pipeline needs to know
which SDK this module uses internally.
"""
import asyncio
import os
import tempfile

from docling.document_converter import DocumentConverter
from llama_cloud_services import LlamaParse

from config import get_settings

# InputFormat.MD's pipeline (SimplePipeline) needs no model config, unlike PDF's
# PdfPipelineOptions - a bare DocumentConverter() is enough, and it's safe to reuse across
# calls (docling converters don't hold per-conversion state between convert() calls).
_MD_CONVERTER = DocumentConverter()

_parser = None


def _get_parser() -> LlamaParse:
    global _parser
    if _parser is None:
        api_key = get_settings().LLAMAPARSE_API_KEY
        if not api_key:
            raise RuntimeError(
                "LLAMAPARSE_API_KEY is not set - add it to Server/analyzerEngine/.env"
            )
        _parser = LlamaParse(api_key=api_key, result_type="markdown")
    return _parser


async def _fetch_result_json(file_path: str) -> dict:
    result = await _get_parser().aparse(file_path)
    return await result.aget_json()


def parse_pdf_pages(file_path: str) -> tuple:
    """Returns (pages, errors). `pages` is a list of (page_no, DoclingDocument) tuples - one
    entry per page LlamaParse returned, in order, page_no 1-indexed as LlamaParse reports it.
    `errors` is the same flat-string-list shape ingestion/docling_utils.conversion_errors()
    uses elsewhere in this package, so callers don't need to special-case which parser
    produced them."""
    try:
        result_json = asyncio.run(_fetch_result_json(file_path))
    except Exception as exc:
        return [], [f"LlamaParse request failed: {exc}"]

    raw_pages = result_json.get("pages") or []
    pages = []
    errors = []
    for i, raw_page in enumerate(raw_pages):
        page_no = raw_page.get("page", i + 1)
        markdown = (raw_page.get("md") or "").strip()
        if not markdown:
            continue
        try:
            pages.append((page_no, _markdown_to_docling(markdown)))
        except Exception as exc:
            errors.append(f"docling markdown parse failed for page {page_no}: {exc}")

    if not pages and not errors:
        errors.append("LlamaParse returned no pages")

    return pages, errors


def _markdown_to_docling(markdown: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(markdown)
        tmp_path = tmp.name
    try:
        return _MD_CONVERTER.convert(tmp_path).document
    finally:
        os.remove(tmp_path)
