from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Hypothesis:
    statement: str
    suggested_investigation: str
    suggested_agent: Literal["tabular", "document", "both"]
    priority: int


@dataclass
class HypothesisResult:
    hypotheses: list = field(default_factory=list)
