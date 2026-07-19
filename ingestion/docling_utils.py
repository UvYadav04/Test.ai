from docling.datamodel.base_models import ConversionStatus


def conversion_errors(result) -> list:
    """Turn a docling ConversionResult's status/errors into the flat string list ingestors
    report on IngestionResult.errors. Shared by every ingestor that runs a file through
    docling's DocumentConverter (pdf, txt - docling auto-detects .txt as markdown)."""
    errors = []
    if result.status != ConversionStatus.SUCCESS:
        errors.append(f"docling conversion status: {result.status.value}")

    for item in result.errors:
        page = f" (page {item.page_no})" if item.page_no else ""
        errors.append(f"docling {item.module_name}{page}: {item.error_message}")

    return errors
