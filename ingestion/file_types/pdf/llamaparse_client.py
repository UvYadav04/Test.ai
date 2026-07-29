import asyncio
import os
import tempfile

from docling.document_converter import DocumentConverter
from llama_cloud_services import LlamaParse

from config import get_settings

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
