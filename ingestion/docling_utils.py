from docling.datamodel.base_models import ConversionStatus


def conversion_errors(result) -> list:
    errors = []
    if result.status != ConversionStatus.SUCCESS:
        errors.append(f"docling conversion status: {result.status.value}")

    for item in result.errors:
        page = f" (page {item.page_no})" if item.page_no else ""
        errors.append(f"docling {item.module_name}{page}: {item.error_message}")

    return errors
