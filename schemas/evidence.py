from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum


class SourceType(str, Enum):
    COMPANIES_HOUSE = "companies_house"
    SEARCH_SNIPPET = "search_snippet"
    NEWS = "news"
    REVIEW_SNIPPET = "review_snippet"


class Evidence(BaseModel):
    source_type: SourceType
    source_url: Optional[str] = None
    title: str
    content: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class EvidenceCollection(BaseModel):
    """Accumulated evidence gathered by the agent pipeline."""

    items: list[Evidence] = Field(default_factory=list)
    queries_executed: list[str] = Field(default_factory=list)
    sources_consulted: list[str] = Field(default_factory=list)
    errors_encountered: list[str] = Field(default_factory=list)

    def add(self, evidence: Evidence) -> None:
        self.items.append(evidence)
        if evidence.source_type.value not in self.sources_consulted:
            self.sources_consulted.append(evidence.source_type.value)

    def record_query(self, query: str) -> None:
        self.queries_executed.append(query)

    def has_query(self, query: str) -> bool:
        return query in self.queries_executed

    def record_error(self, error: str) -> None:
        self.errors_encountered.append(error)

    def by_source(self, source_type: SourceType) -> list[Evidence]:
        return [e for e in self.items if e.source_type == source_type]

    @property
    def count(self) -> int:
        return len(self.items)
