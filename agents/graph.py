"""LangGraph pipeline for underwriting intelligence.

Constructs a linear StateGraph:
  fetch_companies_house → generate_queries → execute_searches
    → summarize_searches → synthesize_report
"""

from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    execute_searches,
    fetch_companies_house,
    generate_queries,
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
    builder.add_node("synthesize_report", synthesize_report)

    builder.add_edge(START, "fetch_companies_house")
    builder.add_edge("fetch_companies_house", "generate_queries")
    builder.add_edge("generate_queries", "execute_searches")
    builder.add_edge("execute_searches", "summarize_searches")
    builder.add_edge("summarize_searches", "synthesize_report")
    builder.add_edge("synthesize_report", END)

    return builder.compile()


pipeline = build_pipeline()
