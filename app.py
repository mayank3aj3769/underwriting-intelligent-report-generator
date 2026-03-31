"""Streamlit entrypoint for the Underwriting Intelligence System.

Single-process deployment: calls the agent pipeline directly,
no separate FastAPI backend required.

Run locally:  streamlit run app.py --server.port 8080
"""

import asyncio
import logging
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

from config import settings
from services.report_generator import ReportGenerator, ReportGenerationResult
from schemas.report import (
    UnderwritingReport as _UnderwritingReport,
    BusinessModelSummary as _BMS,
    CompetitiveLandscape as _CL,
    CompanyQualitySignals as _CQS,
    UncertaintyFlags as _UF,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)


# ── Pydantic models (client-side mirrors for rendering) ──


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


class BusinessModelSummary(BaseModel):
    description: str
    revenue_model: Optional[str] = None
    key_products_services: list[str] = Field(default_factory=list)
    customer_segments: list[str] = Field(default_factory=list)
    geographies: list[str] = Field(default_factory=list)
    evidence_basis: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)


class Competitor(BaseModel):
    name: str
    description: Optional[str] = None
    relevance: Optional[str] = None
    source: Optional[str] = None


class CompetitiveLandscape(BaseModel):
    industry: str
    sic_codes: list[str] = Field(default_factory=list)
    market_position: Optional[str] = None
    competitors: list[Competitor] = Field(default_factory=list)
    competition_degree: CompetitionDegree
    competitive_advantages: list[str] = Field(default_factory=list)
    competitive_disadvantages: list[str] = Field(default_factory=list)
    reasoning: str
    evidence_basis: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)


class QualitySignal(BaseModel):
    signal: str
    sentiment: str
    strength: SignalStrength = SignalStrength.MODERATE
    source: str
    url: Optional[str] = None
    detail: Optional[str] = None


class CompanyQualitySignals(BaseModel):
    signals: list[QualitySignal] = Field(default_factory=list)
    positive_count: int = 0
    negative_count: int = 0
    confidence: ConfidenceLevel
    signal_coverage_assessment: Optional[str] = None
    data_gaps: list[str] = Field(default_factory=list)
    conflicting_signals: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class UncertaintyFlags(BaseModel):
    missing_data: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)
    low_confidence_areas: list[str] = Field(default_factory=list)


class UnderwritingReport(BaseModel):
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


class CandidateCompany(BaseModel):
    company_number: str
    company_name: str
    company_status: Optional[str] = None
    company_type: Optional[str] = None
    date_of_creation: Optional[str] = None
    registered_office_address: Optional[dict] = None
    snippet: Optional[str] = None


# ── Singleton generator (cached across Streamlit reruns) ──


@st.cache_resource
def get_generator() -> ReportGenerator:
    return ReportGenerator()


def run_async(coro):
    """Bridge async pipeline calls into synchronous Streamlit context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Direct pipeline calls (no HTTP) ──


def generate_report(identifier: str) -> ReportGenerationResult:
    gen = get_generator()
    return run_async(gen.generate(identifier))


def generate_report_by_number(company_number: str) -> ReportGenerationResult:
    gen = get_generator()
    return run_async(gen.generate_by_number(company_number))


# ── Rendering helpers ──


def render_report(report: UnderwritingReport):
    st.markdown(f"### {report.company_name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Company Number", report.company_number)
    col2.metric("Evidence Items", report.raw_evidence_count)
    col3.metric("Confidence", report.quality_signals.confidence.value.title())

    st.caption(
        f"Generated: {report.report_generated_at}  |  "
        f"Sources: {', '.join(report.sources_used)}"
    )

    st.divider()

    # Section A: Business Model
    st.markdown("#### A. Business Model Summary")
    st.markdown("*What does this company do? How does it make money? Who are its customers?*")
    st.write(report.business_model.description)
    if report.business_model.revenue_model:
        st.markdown(f"**Revenue Model:** {report.business_model.revenue_model}")
    if report.business_model.key_products_services:
        st.markdown(
            f"**Key Products/Services:** "
            f"{', '.join(report.business_model.key_products_services)}"
        )
    if report.business_model.customer_segments:
        st.markdown(
            f"**Customer Segments:** {', '.join(report.business_model.customer_segments)}"
        )
    if report.business_model.geographies:
        st.markdown(
            f"**Geographies:** {', '.join(report.business_model.geographies)}"
        )
    if report.business_model.evidence_basis:
        st.caption(f"Evidence basis: {report.business_model.evidence_basis}")

    st.divider()

    # Section B: Competitive Landscape
    st.markdown("#### B. Competitive Landscape")
    st.markdown(
        "*Structured assessment of competitive intensity, "
        "specific to sector and geography.*"
    )
    st.markdown(
        f"**Industry:** {report.competitive_landscape.industry}  |  "
        f"**Competition:** {report.competitive_landscape.competition_degree.value.title()}"
    )
    if report.competitive_landscape.market_position:
        st.markdown(f"**Market Position:** {report.competitive_landscape.market_position}")

    st.markdown(f"\n{report.competitive_landscape.reasoning}")

    if report.competitive_landscape.competitors:
        st.markdown("**Key Competitors:**")
        for comp in report.competitive_landscape.competitors:
            desc = f" -- {comp.description}" if comp.description else ""
            rel = f"  \n*Relevance: {comp.relevance}*" if comp.relevance else ""
            st.markdown(f"- **{comp.name}**{desc}{rel}")

    if report.competitive_landscape.competitive_advantages:
        with st.container():
            st.markdown("**Competitive Advantages:**")
            for adv in report.competitive_landscape.competitive_advantages:
                st.markdown(f"- :green[+] {adv}")

    if report.competitive_landscape.competitive_disadvantages:
        with st.container():
            st.markdown("**Competitive Disadvantages:**")
            for dis in report.competitive_landscape.competitive_disadvantages:
                st.markdown(f"- :red[-] {dis}")

    if report.competitive_landscape.evidence_basis:
        st.caption(f"Evidence basis: {report.competitive_landscape.evidence_basis}")

    st.divider()

    # Section C: Quality Signals
    st.markdown("#### C. Company Quality Signals")
    st.markdown(
        "*Structured synthesis of quality indicators from reviews, "
        "trade press, and public sources.*"
    )

    pos = report.quality_signals.positive_count
    neg = report.quality_signals.negative_count
    neu = len(report.quality_signals.signals) - pos - neg
    st.markdown(
        f"Positive: **{pos}** | Negative: **{neg}** | Neutral: **{neu}**  \n"
        f"Overall confidence: **{report.quality_signals.confidence.value}**"
    )

    if report.quality_signals.signal_coverage_assessment:
        st.info(report.quality_signals.signal_coverage_assessment)

    for sig in report.quality_signals.signals:
        icon = {"positive": ":green[+]", "negative": ":red[-]", "neutral": ":orange[~]"}.get(
            sig.sentiment, ":orange[~]"
        )
        strength_badge = {
            "strong": ":green[strong]",
            "moderate": ":orange[moderate]",
            "weak": ":red[weak]",
        }.get(sig.strength.value, ":orange[moderate]")

        url_part = f" ([source]({sig.url}))" if sig.url else f" *(source: {sig.source})*"
        st.markdown(f"- {icon} **{sig.signal}** [{strength_badge}]{url_part}")
        if sig.detail:
            st.caption(f"  > {sig.detail}")

    if report.quality_signals.data_gaps:
        st.markdown("**Data Gaps** (where evidence is thin or absent):")
        for gap in report.quality_signals.data_gaps:
            st.markdown(f"- :orange[!] {gap}")

    if report.quality_signals.conflicting_signals:
        st.markdown("**Conflicting Signals:**")
        for conflict in report.quality_signals.conflicting_signals:
            st.markdown(f"- :red[?] {conflict}")

    st.divider()

    # Outlook
    if report.business_outlook:
        st.markdown("#### D. Business Outlook")
        st.write(report.business_outlook)
    if report.sectoral_outlook:
        st.markdown("#### E. Sectoral Outlook")
        st.write(report.sectoral_outlook)
    if report.business_outlook or report.sectoral_outlook:
        st.divider()

    # Uncertainty
    flags = report.uncertainty_flags
    has_flags = flags.missing_data or flags.conflicting_evidence or flags.low_confidence_areas
    st.markdown("#### F. Uncertainty Flags")
    st.markdown("*The system surfaces uncertainty rather than papering over it.*")
    if has_flags:
        if flags.missing_data:
            st.markdown("**Missing data:**")
            for item in flags.missing_data:
                st.markdown(f"- {item}")
        if flags.conflicting_evidence:
            st.markdown("**Conflicting evidence:**")
            for item in flags.conflicting_evidence:
                st.markdown(f"- {item}")
        if flags.low_confidence_areas:
            st.markdown("**Low confidence areas:**")
            for item in flags.low_confidence_areas:
                st.markdown(f"- {item}")
    else:
        st.markdown("No significant uncertainty flags.")

    st.divider()

    # Sources
    all_citations = report.business_model.citations
    if all_citations:
        with st.expander(f"G. Sources & Citations ({len(all_citations)})", expanded=False):
            for cit in all_citations:
                label = cit.detail or cit.source
                if cit.url:
                    st.markdown(f"- [{label}]({cit.url})")
                else:
                    st.markdown(f"- {label} *({cit.source})*")


def render_disambiguation(candidates: list[CandidateCompany], query: str):
    st.markdown(
        f"I found multiple companies matching **\"{query}\"**. "
        "Which one did you mean?"
    )
    st.markdown("")

    for c in candidates[:10]:
        status_badge = {
            "active": ":green[active]",
            "dissolved": ":red[dissolved]",
        }.get(c.company_status or "", f":orange[{c.company_status}]")

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(
                f"**{c.company_name}**  \n"
                f"`{c.company_number}` | {status_badge}"
                + (f" | Est. {c.date_of_creation}" if c.date_of_creation else "")
            )
        with col2:
            if st.button("Select", key=f"btn_{c.company_number}"):
                st.session_state.messages.append(
                    {"role": "user", "content": f"Generate report for {c.company_number}"}
                )
                st.session_state.pending_number = c.company_number
                st.rerun()


# ── Main app ──


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_number" not in st.session_state:
        st.session_state.pending_number = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def main():
    st.set_page_config(
        page_title="Underwriting Intelligence",
        page_icon="shield",
        layout="wide",
    )

    st.markdown(
        "<h2 style='text-align: center;'>Underwriting Intelligence Report Generator</h2>"
        "<p style='text-align: center; color: grey;'>"
        "Enter a UK company name or Companies House number to generate a report."
        "</p>",
        unsafe_allow_html=True,
    )

    init_session()

    # Sidebar
    with st.sidebar:
        st.markdown("### System Status")
        keys_ok = (
            settings.has_companies_house_key
            and settings.has_serp_api_key
            and settings.has_openai_key
        )
        if keys_ok:
            st.markdown("System: :green[ready]")
            st.markdown("Pipeline: `langgraph`")
            st.markdown("Search: `serpapi`")
            st.markdown("Synthesis: `openai`")
        else:
            missing = []
            if not settings.has_companies_house_key:
                missing.append("COMPANIES_HOUSE_API_KEY")
            if not settings.has_serp_api_key:
                missing.append("SERP_API_KEY")
            if not settings.has_openai_key:
                missing.append("OPENAI_API_KEY")
            st.markdown(":red[Missing API keys]")
            for k in missing:
                st.markdown(f"- `{k}`")

        st.divider()
        st.markdown("### Quick Examples")
        examples = [
            ("REVOLUT LTD", "Exact name"),
            ("08804411", "Company number"),
            ("Barclays", "Disambiguation"),
        ]
        for ex, label in examples:
            if st.button(f"{label}: {ex}", key=f"ex_{ex}"):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.session_state.pending_query = ex
                st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "report" and msg.get("report_data"):
                try:
                    report = UnderwritingReport(**msg["report_data"])
                    render_report(report)
                except ValidationError as e:
                    st.error(f"Report parsing failed: {e}")
            elif msg.get("type") == "disambiguation" and msg.get("candidates"):
                candidates = [CandidateCompany(**c) for c in msg["candidates"]]
                render_disambiguation(candidates, msg.get("query", ""))
            else:
                st.markdown(msg["content"])

    # Handle pending company number from disambiguation
    if st.session_state.pending_number:
        number = st.session_state.pending_number
        st.session_state.pending_number = None
        _generate_by_number(number)
        st.rerun()

    # Handle pending query from sidebar
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        _handle_query(query)
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Enter company name or number..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        _handle_query(prompt)


def _handle_query(query: str):
    """Process a user query by calling the pipeline directly."""
    with st.chat_message("assistant"):
        with st.spinner("Generating intelligence report... this may take up to 2 minutes."):
            try:
                result = generate_report(query)
            except Exception as e:
                msg = f"Pipeline error: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                return

        if result.needs_disambiguation and result.disambiguation:
            candidates = []
            for c in result.disambiguation.candidates[:10]:
                try:
                    candidates.append(CandidateCompany(**c.model_dump()))
                except ValidationError:
                    continue

            render_disambiguation(candidates, query)
            st.session_state.messages.append({
                "role": "assistant",
                "type": "disambiguation",
                "content": f"Multiple matches for \"{query}\".",
                "candidates": [c.model_dump() for c in candidates],
                "query": query,
            })
            return

        if result.is_error:
            st.error(result.error or "Unknown error")
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.error or "Unknown error",
            })
            return

        if result.report:
            try:
                report_data = result.report.model_dump()
                report = UnderwritingReport(**report_data)
            except ValidationError as e:
                msg = f"Report validation failed: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                return

            render_report(report)
            st.session_state.messages.append({
                "role": "assistant",
                "type": "report",
                "content": f"Report generated for {report.company_name}",
                "report_data": report_data,
            })
            return

        st.warning("No result returned from pipeline.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "No result returned from pipeline.",
        })


def _generate_by_number(company_number: str):
    """Generate a report directly by company number (from disambiguation)."""
    with st.chat_message("assistant"):
        with st.spinner("Generating report... this may take up to 2 minutes."):
            try:
                result = generate_report_by_number(company_number)
            except Exception as e:
                msg = f"Pipeline error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                return

        if result.report:
            try:
                report_data = result.report.model_dump()
                report = UnderwritingReport(**report_data)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "report",
                    "content": f"Report generated for {report.company_name}",
                    "report_data": report_data,
                })
            except ValidationError as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Report validation failed: {e}",
                })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.error or "Report generation failed.",
            })


if __name__ == "__main__":
    main()
