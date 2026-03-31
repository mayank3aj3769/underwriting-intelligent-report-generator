from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CompetitionDegree(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SignalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class Citation(BaseModel):
    source: str
    url: Optional[str] = None
    detail: Optional[str] = None


# ── Section 1: Business Model ──


class BusinessModelSummary(BaseModel):
    description: str = Field(
        ..., description="What the company actually does — grounded in evidence"
    )
    revenue_model: Optional[str] = Field(
        None, description="How it generates revenue — subscription, transaction fees, etc."
    )
    current_revenue: Optional[str] = Field(
        None,
        description=(
            "Most recent publicly available revenue figure with year/period, "
            "or explicit statement that revenue data is not publicly available"
        ),
    )
    revenue_trend: Optional[str] = Field(
        None,
        description=(
            "Revenue trajectory — growth/decline direction, rate, and time period "
            "(e.g. '35% YoY growth in 2024'), or explicit statement that trend data "
            "is not publicly available"
        ),
    )
    key_products_services: list[str] = Field(
        default_factory=list, description="Core product/service offerings"
    )
    customer_segments: list[str] = Field(
        default_factory=list, description="Who the customers are"
    )
    geographies: list[str] = Field(
        default_factory=list, description="Key markets/regions served"
    )
    evidence_basis: Optional[str] = Field(
        None,
        description="Summary of which sources informed this section and how reliable they are",
    )
    citations: list[Citation] = Field(default_factory=list)


# ── Section 2: Competitive Landscape ──


class Competitor(BaseModel):
    name: str
    description: Optional[str] = None
    relevance: Optional[str] = Field(
        None, description="Why this competitor is relevant to the subject company"
    )
    source: Optional[str] = None


class CompetitiveLandscape(BaseModel):
    industry: str
    sic_codes: list[str] = Field(default_factory=list)
    market_position: Optional[str] = Field(
        None,
        description="Where the company sits in the market — leader, challenger, niche, etc.",
    )
    competitors: list[Competitor] = Field(default_factory=list)
    competition_degree: CompetitionDegree
    competitive_advantages: list[str] = Field(
        default_factory=list, description="What gives this company an edge"
    )
    competitive_disadvantages: list[str] = Field(
        default_factory=list, description="Where this company is weaker vs peers"
    )
    reasoning: str = Field(
        ...,
        description=(
            "Evidence-backed analysis of competitive intensity — "
            "not just a label, but WHY that degree of competition matters "
            "for underwriting this business"
        ),
    )
    evidence_basis: Optional[str] = Field(
        None,
        description="Summary of which sources informed this section and how reliable they are",
    )
    citations: list[Citation] = Field(default_factory=list)


# ── Section 3: Quality Signals ──


class QualitySignal(BaseModel):
    signal: str = Field(..., description="The observed quality indicator")
    sentiment: str  # positive / negative / neutral
    strength: SignalStrength = Field(
        default=SignalStrength.MODERATE,
        description="How strong/reliable is this individual signal",
    )
    source: str
    url: Optional[str] = None
    detail: Optional[str] = Field(
        None, description="Supporting quote or data point from the source"
    )


class CompanyQualitySignals(BaseModel):
    signals: list[QualitySignal] = Field(default_factory=list)
    positive_count: int = 0
    negative_count: int = 0
    confidence: ConfidenceLevel
    signal_coverage_assessment: Optional[str] = Field(
        None,
        description=(
            "Overall assessment of evidence quality — is there enough data "
            "to be confident? Which signal categories are well-covered vs thin?"
        ),
    )
    data_gaps: list[str] = Field(
        default_factory=list,
        description="Specific areas where evidence is thin, absent, or conflicting",
    )
    conflicting_signals: list[str] = Field(
        default_factory=list,
        description="Pairs or groups of signals that contradict each other",
    )
    missing_data: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


# ── Uncertainty ──


class UncertaintyFlags(BaseModel):
    missing_data: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)
    low_confidence_areas: list[str] = Field(default_factory=list)


# ── Top-level Report ──


class UnderwritingReport(BaseModel):
    """Complete underwriting intelligence report."""

    company_name: str
    company_number: str
    report_generated_at: str
    sources_used: list[str] = Field(default_factory=list)
    business_model: BusinessModelSummary
    competitive_landscape: CompetitiveLandscape
    quality_signals: CompanyQualitySignals
    uncertainty_flags: UncertaintyFlags
    business_outlook: Optional[str] = None
    sectoral_outlook: Optional[str] = None
    raw_evidence_count: int = 0
    readable_report: Optional[str] = None

    evidence_confidence_score: float = Field(
        default=0.0,
        description="Confidence score from the evidence sufficiency evaluation (0.0–1.0)",
    )
    evidence_iterations: int = Field(
        default=1,
        description="Number of evidence-gathering iterations the agent performed",
    )
    evidence_gaps_found: list[str] = Field(
        default_factory=list,
        description="Evidence gaps identified during sufficiency evaluation",
    )
