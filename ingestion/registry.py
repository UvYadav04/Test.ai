"""Maps file extensions to ingestor classes.

Adding a new file type = add a new folder under ingestion/file_types/ with
an ingestor implementing BaseIngestor, then register its extension(s) here.
The manager never branches on file type itself -- it always goes through
get_ingestor_for().
"""
import os
from typing import Type

from ingestion.file_types.base import BaseIngestor
from ingestion.file_types.csv.csv_ingestor import CSVIngestor
from ingestion.file_types.json.json_ingestor import JSONIngestor
from ingestion.file_types.pdf.pdf_ingestor import PDFIngestor

EXTENSION_REGISTRY: dict[str, Type[BaseIngestor]] = {
    ".csv": CSVIngestor,
    ".json": JSONIngestor,
    ".pdf": PDFIngestor,
}


def get_ingestor_for(file_path: str) -> Type[BaseIngestor]:

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    ingestor_cls = EXTENSION_REGISTRY.get(ext)
    if ingestor_cls is None:
        supported = ", ".join(sorted(EXTENSION_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported file type '{ext or '<none>'}' for '{file_path}'. "
            f"Supported types: {supported}"
        )
    return ingestor_cls
