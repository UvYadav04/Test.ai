from docling.document_converter import DocumentConverter


def convert_document(file_path: str):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document


def get_page_count(document) -> int:
    return len(document.pages)


def is_scanned(document) -> bool:
    text = document.export_to_text()
    return len(text.strip()) < 20
