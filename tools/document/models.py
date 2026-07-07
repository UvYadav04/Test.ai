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
