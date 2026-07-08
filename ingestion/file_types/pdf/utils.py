from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def convert_document(file_path: str) -> tuple:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(file_path)
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


def extract_tables(document) -> list:
    tables = []
    for index, table in enumerate(document.tables):
        try:
            dataframe = table.export_to_dataframe(doc=document)
        except Exception:
            continue
        if dataframe.empty:
            continue
        tables.append({
            "index": index,
            "dataframe": dataframe,
            "page": _table_page(table),
            "caption": _table_caption(table, document),
        })
    return tables


def _table_page(table) -> int:
    try:
        return table.prov[0].page_no
    except Exception:
        return 0


def _table_caption(table, document) -> str:
    try:
        return table.caption_text(document) or ""
    except Exception:
        return ""
