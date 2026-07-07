from dataclasses import dataclass, field


@dataclass
class ChunkRecord:
    chunk_id: str
    file_id: str
    workspace_id: str
    text: str
    metadata: dict = field(default_factory=dict)
