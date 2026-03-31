"""LangGraph pipeline for underwriting intelligence.

Constructs an adaptive StateGraph with an evidence sufficiency loop:

  fetch_companies_house → generate_queries → execute_searches
    → summarize_searches → evaluate_sufficiency
    ─── if sufficient ──────────────────────→ synthesize_report
    ─── if insufficient → generate_gap_queries → execute_searches
          → summarize_searches → evaluate_sufficiency (max 2 iterations)
"""

from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    evaluate_sufficiency,
    execute_searches,
    fetch_companies_house,
    generate_gap_queries,
    generate_queries,
    sufficiency_router,
    summarize_searches,
    synthesize_report,
)
from agents.state import PipelineState


def build_pipeline() -> StateGraph:
    """Construct and compile the underwriting intelligence graph."""
    builder = StateGraph(PipelineState)

    builder.add_node("fetch_companies_house", fetch_companies_house)
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("execute_searches", execute_searches)
    builder.add_node("summarize_searches", summarize_searches)
    builder.add_node("evaluate_sufficiency", evaluate_sufficiency)
    builder.add_node("generate_gap_queries", generate_gap_queries)
    builder.add_node("synthesize_report", synthesize_report)

    builder.add_edge(START, "fetch_companies_house")
    builder.add_edge("fetch_companies_house", "generate_queries")
    builder.add_edge("generate_queries", "execute_searches")
    builder.add_edge("execute_searches", "summarize_searches")
    builder.add_edge("summarize_searches", "evaluate_sufficiency")

    builder.add_conditional_edges(
        "evaluate_sufficiency",
        sufficiency_router,
        {
            "synthesize_report": "synthesize_report",
            "generate_gap_queries": "generate_gap_queries",
        },
    )

    builder.add_edge("generate_gap_queries", "execute_searches")
    builder.add_edge("synthesize_report", END)

    return builder.compile()


pipeline = build_pipeline()
