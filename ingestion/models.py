from dataclasses import dataclass, field
from typing import Literal, Optional

IngestionStatus = Literal["success", "partial", "failed"]


@dataclass
class IngestionResult:
    file_id: str
    workspace_id: str
    status: IngestionStatus
    output_ref: str
    schema_summary: dict
    row_count: Optional[int] = None
    chunk_count: Optional[int] = None
    extracted_tables: list = field(default_factory=list)
    errors: list = field(default_factory=list)
