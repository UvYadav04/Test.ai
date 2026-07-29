from docling.document_converter import DocumentConverter

from ingestion.docling_utils import conversion_errors


def convert_document(file_path: str) -> tuple:
    result = DocumentConverter().convert(file_path)
    return result.document, conversion_errors(result)
