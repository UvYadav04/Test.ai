from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    chunk_id: str
    file_id: str
    text: str
    score: float
    metadata: dict


@dataclass
class SectionInfo:
    section_title: str
    page_start: int
    page_end: int


@dataclass
class ComparisonResult:
    per_file_findings: dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    supported: bool
    reasoning: str


@dataclass
class TableInfo:
    table_ref: str
    page: int
    caption: str
    columns: list
    row_count: int


@dataclass
class FileOverview:
    file_id: str
    sections: list
    tables: list
    key_topics: list
