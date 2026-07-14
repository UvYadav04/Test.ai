from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def convert_document(file_path: str) -> tuple:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_page_images = False
    pipeline_options.images_scale = 0.5

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(file_path,page_range=(1,5))
    return result.document, _conversion_errors(result)


def _conversion_errors(result) -> list:
    errors = []
    if result.status != ConversionStatus.SUCCESS:
        errors.append(f"docling conversion status: {result.status.value}")

    for item in result.errors:
        page = f" (page {item.page_no})" if item.page_no else ""
        errors.append(f"docling {item.module_name}{page}: {item.error_message}")

    return errors


def get_page_count(document) -> int:
    return len(document.pages)


def is_scanned(document) -> bool:
    text = document.export_to_text()
    return len(text.strip()) < 20


def extract_tables(document, chunks: list = None) -> list:
    tables = []
    for index, table in enumerate(document.tables):
        try:
            dataframe = table.export_to_dataframe(doc=document)
        except Exception:
            continue
        if dataframe.empty:
            continue
        page = _table_page(table)
        tables.append({
            "index": index,
            "dataframe": dataframe,
            "page": page,
            "caption": _infer_caption(table, document, chunks, page),
        })
    return tables


def _table_page(table) -> int:
    try:
        return table.prov[0].page_no
    except Exception:
        return 0


def _infer_caption(table, document, chunks: list, page: int) -> str:
    """4-tier fallback: explicit docling caption -> nearby text snippet -> section title -> none."""
    explicit = _explicit_caption(table, document)
    if explicit:
        return explicit

    nearby = _nearest_chunk(chunks, page)
    if nearby is None:
        return f"Table on page {page}"

    snippet = nearby.text.strip().splitlines()[0][:120] if nearby.text.strip() else ""
    if snippet:
        return snippet

    if nearby.section:
        return nearby.section

    return f"Table on page {page}"


def _explicit_caption(table, document) -> str:
    try:
        return table.caption_text(document) or ""
    except Exception:
        return ""


def _nearest_chunk(chunks: list, page: int):
    if not chunks:
        return None
    before = [c for c in chunks if c.page <= page]
    pool = before or chunks
    return max(pool, key=lambda c: c.page)
