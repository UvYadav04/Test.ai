from docling.document_converter import DocumentConverter

from ingestion.docling_utils import conversion_errors


def convert_document(file_path: str) -> tuple:
    """Plain text needs none of PDF's pipeline options (no OCR, no page images) - docling
    auto-detects .txt as InputFormat.MD and runs it through SimplePipeline/
    MarkdownDocumentBackend by default, so a plain `DocumentConverter().convert()` is enough.
    This gives TXTIngestor the same docling `document` shape PDFIngestor gets, so it can reuse
    DoclingChunker (HybridChunker) instead of a bespoke text splitter."""
    result = DocumentConverter().convert(file_path)
    return result.document, conversion_errors(result)
