from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class FileRef:
    file_id: str
    output_ref: str
    filename: str = ""


@dataclass
class FileMetadata:
    file_id: str
    filename: str
    output_ref: str
    row_count: int
    columns: list


@dataclass
class SchemaInfo:
    columns: list
    dtypes: dict
    nullable: dict
    sample_size: int
    likely_key_columns: list


@dataclass
class QueryResult:
    columns: list
    rows: list
    row_count: int
    truncated: bool
    error: Optional[str] = None


@dataclass
class MetricSpec:
    column: str
    op: Literal["sum", "avg", "count", "min", "max"]
    alias: Optional[str] = None


@dataclass
class JoinCandidate:
    file_a: str
    column_a: str
    file_b: str
    column_b: str
    match_confidence: float


@dataclass
class ColumnProfile:
    min: Any
    max: Any
    mean: Optional[float]
    null_count: int
    distinct_count: int
    top_values: list


@dataclass
class ValidationReport:
    passed: bool
    warnings: list = field(default_factory=list)
