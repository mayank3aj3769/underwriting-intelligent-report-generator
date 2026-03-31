"""Underwriting Intelligence Report API.

FastAPI application that generates structured underwriting intelligence
reports for UK companies using an adaptive LangGraph agent pipeline.

Pipeline: CH Fetch → Query Gen → Search → Summarise → Evaluate Sufficiency
          ↳ (if insufficient) → Gap Queries → Re-search → Re-summarise → Re-evaluate
          → Synthesise Report
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import settings
from services.entity_resolver import EntityResolver
from services.report_generator import ReportGenerator, ReportGenerationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

report_generator = ReportGenerator()
entity_resolver = EntityResolver()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Underwriting Intelligence API starting up")
    missing: list[str] = []
    if not settings.has_companies_house_key:
        missing.append("COMPANIES_HOUSE_API_KEY")
    if not settings.has_serp_api_key:
        missing.append("SERP_API_KEY")
    if not settings.has_openai_key:
        missing.append("OPENAI_API_KEY")
    if missing:
        logger.error("MISSING REQUIRED KEYS: %s", ", ".join(missing))
    else:
        logger.info("All API keys configured — system ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Underwriting Intelligence API",
    description=(
        "Generates structured underwriting intelligence reports for UK companies "
        "using a LangGraph pipeline with LLM-driven search and synthesis. "
        "All searches via SerpAPI. All synthesis via OpenAI."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# ---- Request / response models ----

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Company name to search for")


class ReportRequest(BaseModel):
    identifier: str = Field(
        ...,
        min_length=1,
        description="Company name or Companies House registration number",
    )


class ReportByNumberRequest(BaseModel):
    company_number: str = Field(
        ...,
        min_length=1,
        description="Companies House registration number",
    )


class FollowUpRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Follow-up question about the report")
    report_context: dict = Field(
        ...,
        description="Report context containing company_name, company_number, and readable_report",
    )


# ---- Endpoints ----

@app.get("/health")
async def health_check():
    all_ok = (
        settings.has_companies_house_key
        and settings.has_serp_api_key
        and settings.has_openai_key
    )
    return {
        "status": "ok" if all_ok else "degraded",
        "companies_house_configured": settings.has_companies_house_key,
        "serp_api_configured": settings.has_serp_api_key,
        "openai_configured": settings.has_openai_key,
        "pipeline": "langgraph",
        "search_backend": "serpapi",
        "synthesis_backend": "openai",
    }


@app.post("/api/v1/search")
async def search_companies(request: SearchRequest):
    """Search for UK companies by name.

    Returns a ranked list of candidates from Companies House.
    """
    if not settings.has_companies_house_key:
        raise HTTPException(
            status_code=503,
            detail="Companies House API key not configured",
        )

    result = await entity_resolver.search_by_name(request.query)
    return {
        "query": result.query,
        "total_results": result.total_results,
        "disambiguation_required": result.disambiguation_required,
        "candidates": [c.model_dump() for c in result.candidates],
    }


@app.post("/api/v1/report")
async def generate_report(request: ReportRequest):
    """Generate an underwriting intelligence report.

    Accepts a company name or Companies House number.
    If the name is ambiguous, returns a disambiguation response.
    """
    _require_all_keys()
    result = await report_generator.generate(request.identifier)
    return _format_result(result)


@app.post("/api/v1/report/by-number")
async def generate_report_by_number(request: ReportByNumberRequest):
    """Generate a report directly from a Companies House number."""
    _require_all_keys()
    result = await report_generator.generate_by_number(request.company_number)
    return _format_result(result)


@app.post("/api/v1/report/follow-up")
async def follow_up_question(request: FollowUpRequest):
    """Ask a follow-up question about a previously generated report."""
    _require_all_keys()
    result = await report_generator.generate(
        request.question,
        report_context=request.report_context,
    )
    return _format_result(result)


def _require_all_keys() -> None:
    missing: list[str] = []
    if not settings.has_companies_house_key:
        missing.append("COMPANIES_HOUSE_API_KEY")
    if not settings.has_serp_api_key:
        missing.append("SERP_API_KEY")
    if not settings.has_openai_key:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Required API keys not configured: {', '.join(missing)}",
        )


def _format_result(result: ReportGenerationResult) -> dict:
    data = result.to_dict()
    if result.is_rejected:
        raise HTTPException(status_code=422, detail=data["message"])
    if result.is_error:
        raise HTTPException(status_code=404, detail=data["message"])
    return data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
