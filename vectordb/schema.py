from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkRecord:
    chunk_id: str
    file_id: str
    workspace_id: str
    text: str
    embedding: Optional[list] = None
    metadata: dict = field(default_factory=dict)
