"""LangGraph pipeline state definition."""

import operator
from typing import Annotated, TypedDict


class PipelineState(TypedDict):
    """State that flows through the LangGraph underwriting pipeline.

    Nodes:
      fetch_companies_house → generate_queries → execute_searches
        → summarize_searches → synthesize_report
    """

    # --- Inputs (set before graph invocation) ---
    company_number: str
    company_name: str

    # --- Populated by fetch_companies_house ---
    company_profile_text: str
    company_metadata: dict

    # --- Populated by generate_queries ---
    search_queries: list[str]

    # --- Populated by execute_searches ---
    search_results: list[dict]

    # --- Populated by summarize_searches ---
    search_summary: str

    # --- Populated by synthesize_report ---
    final_report: dict

    # --- Accumulated across all nodes ---
    errors: Annotated[list[str], operator.add]
