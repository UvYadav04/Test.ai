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


@dataclass
class InvestigationEvent:
    event_type: str  # "tabular" | "document" | "hypothesis"
    objective: str
    result: object
    timestamp: str


@dataclass
class InvestigationState:
    session_id: str
    objective: str
    constraints: dict = field(default_factory=dict)
    selected_files: list = field(default_factory=list)
    active_tasks: list = field(default_factory=list)
    completed_tasks: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    status: str = "in_progress"

    def add_event(self, event: InvestigationEvent) -> None:
        self.completed_tasks.append(event)
        self.findings.append(event.result)

    def summary(self) -> str:
        lines = [f"Investigation for: {self.objective}", f"Status: {self.status}"]
        for event in self.completed_tasks:
            result_summary = getattr(event.result, "summary", str(event.result))
            lines.append(f"- [{event.event_type}] {event.objective} -> {result_summary}")
        if self.open_questions:
            lines.append("Open questions: " + "; ".join(self.open_questions))
        return "\n".join(lines)


@dataclass
class OrchestratorResult:
    final_answer: str
    confidence: str
    artifact_refs: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
