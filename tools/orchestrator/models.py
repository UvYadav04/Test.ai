from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class FileCatalogEntry:
    file_id: str
    filename: str
    file_type: str
    uploaded_at: datetime
    size_bytes: int
    output_ref: str = ""
    row_count: Optional[int] = None
    page_count: Optional[int] = None
    columns: Optional[list] = None
    tags: Optional[list] = None


@dataclass
class FileRef:
    file_id: str
    output_ref: str


@dataclass
class TabularFindings:
    summary: str
    findings: list
    limitations: str
    confidence: str
    artifact_refs: list = field(default_factory=list)


@dataclass
class DocumentFindings:
    summary: str
    findings: list
    limitations: str
    confidence: str
    artifact_refs: list = field(default_factory=list)
    source_refs: list = field(default_factory=list)
