"""LangGraph pipeline state definition."""

import operator
from typing import Annotated, TypedDict


class EvidenceMetrics(TypedDict, total=False):
    """Metrics computed by the evaluate_sufficiency node."""

    total_sources: int
    has_business_info: bool
    has_competitors: bool
    has_reviews: bool
    has_news: bool
    has_financial_signals: bool
    section_coverage: dict[str, bool]
    confidence_score: float
    missing_sections: list[str]
    is_sufficient: bool
    reasoning: str


class PipelineState(TypedDict):
    """State that flows through the LangGraph underwriting pipeline.

    Nodes:
      fetch_companies_house → generate_queries → execute_searches
        → summarize_searches → evaluate_sufficiency
        → (loop if insufficient) → synthesize_report
    """

    # --- Inputs (set before graph invocation) ---
    company_number: str
    company_name: str

    # --- Populated by fetch_companies_house ---
    company_profile_text: str
    company_metadata: dict

    # --- Populated by generate_queries / generate_gap_queries ---
    search_queries: list[str]

    # --- Populated by execute_searches ---
    search_results: list[dict]

    # --- Populated by summarize_searches ---
    search_summary: str

    # --- Populated by evaluate_sufficiency ---
    evidence_metrics: EvidenceMetrics
    iteration_count: int
    sufficiency_flag: bool

    # --- Populated by synthesize_report ---
    final_report: dict

    # --- Accumulated across all nodes ---
    errors: Annotated[list[str], operator.add]
