"""Streamlit chat interface for the Underwriting Intelligence System.

Connects to the FastAPI backend, validates responses with Pydantic,
and renders reports in a conversational chat layout.

Run:  streamlit run streamlit_app.py
"""

import httpx
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from enum import Enum


# ── Pydantic models (mirror schemas/report.py for client-side validation) ──


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


class APIResponse(BaseModel):
    status: str
    message: Optional[str] = None
    report: Optional[dict] = None
    total_results: Optional[int] = None
    candidates: Optional[list[dict]] = None


# ── Config ──

API_BASE = "http://localhost:8000"
TIMEOUT = 300.0


# ── API helpers ──


def call_report_api(identifier: str) -> dict:
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            f"{API_BASE}/api/v1/report",
            json={"identifier": identifier},
        )
        if resp.status_code == 404:
            return {"status": "error", "message": resp.json().get("detail", "Not found")}
        if resp.status_code == 503:
            return {"status": "error", "message": resp.json().get("detail", "Service unavailable")}
        resp.raise_for_status()
        return resp.json()


def call_report_by_number_api(company_number: str) -> dict:
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            f"{API_BASE}/api/v1/report/by-number",
            json={"company_number": company_number},
        )
        if resp.status_code == 404:
            return {"status": "error", "message": resp.json().get("detail", "Not found")}
        if resp.status_code == 503:
            return {"status": "error", "message": resp.json().get("detail", "Service unavailable")}
        resp.raise_for_status()
        return resp.json()


def check_health() -> dict | None:
    try:
        with httpx.Client(timeout=10) as client:
            return client.get(f"{API_BASE}/health").json()
    except Exception:
        return None


# ── Rendering helpers ──


def render_report(report: UnderwritingReport):
    """Render a validated report inside a chat assistant message."""

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

    # ── Section 1: Business Model ──
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

    # ── Section 2: Competitive Landscape ──
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

    # ── Section 3: Quality Signals ──
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

    # ── Outlook ──
    if report.business_outlook:
        st.markdown("#### D. Business Outlook")
        st.write(report.business_outlook)
    if report.sectoral_outlook:
        st.markdown("#### E. Sectoral Outlook")
        st.write(report.sectoral_outlook)
    if report.business_outlook or report.sectoral_outlook:
        st.divider()

    # ── Uncertainty ──
    flags = report.uncertainty_flags
    has_flags = flags.missing_data or flags.conflicting_evidence or flags.low_confidence_areas
    st.markdown("#### F. Uncertainty Flags")
    st.markdown(
        "*The system surfaces uncertainty rather than papering over it.*"
    )
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

    # ── Sources ──
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
    """Render disambiguation options as clickable buttons."""

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

    # Sidebar — system health
    with st.sidebar:
        st.markdown("### System Status")
        health = check_health()
        if health:
            status = health.get("status", "unknown")
            colour = "green" if status == "ok" else "orange"
            st.markdown(f"Backend: :{colour}[{status}]")
            st.markdown(f"Pipeline: `{health.get('pipeline', '?')}`")
            st.markdown(f"Search: `{health.get('search_backend', '?')}`")
            st.markdown(f"Synthesis: `{health.get('synthesis_backend', '?')}`")
        else:
            st.markdown(":red[Backend offline]")
            st.info("Start the API server first:\n\n`uvicorn main:app --reload`")

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

    # Handle pending company number from disambiguation selection
    if st.session_state.pending_number:
        number = st.session_state.pending_number
        st.session_state.pending_number = None
        _generate_report_by_number(number)
        st.rerun()

    # Handle pending query from sidebar quick examples
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
    """Process a user query — call the API and handle the response."""
    with st.chat_message("assistant"):
        with st.spinner("Generating intelligence report... this may take up to 2 minutes."):
            try:
                raw = call_report_api(query)
            except httpx.ConnectError:
                msg = "Cannot connect to the API server. Make sure it is running: `uvicorn main:app --reload`"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                return
            except Exception as e:
                msg = f"API request failed: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                return

        try:
            response = APIResponse(**raw)
        except ValidationError as e:
            msg = f"Invalid API response: {e}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            return

        if response.status == "disambiguation_required" and response.candidates:
            candidates = []
            for c in response.candidates[:10]:
                try:
                    candidates.append(CandidateCompany(**c))
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

        if response.status == "error":
            st.error(response.message or "Unknown error")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.message or "Unknown error",
            })
            return

        if response.status == "success" and response.report:
            try:
                report = UnderwritingReport(**response.report)
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
                "report_data": response.report,
            })
            return

        st.warning(f"Unexpected response status: {response.status}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Unexpected status: {response.status}",
        })


def _generate_report_by_number(company_number: str):
    """Generate a report directly by company number (from disambiguation)."""
    with st.chat_message("assistant"):
        with st.spinner("Generating report... this may take up to 2 minutes."):
            try:
                raw = call_report_by_number_api(company_number)
            except Exception as e:
                msg = f"API request failed: {e}"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                return

        try:
            response = APIResponse(**raw)
        except ValidationError as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Invalid response: {e}",
            })
            return

        if response.status == "success" and response.report:
            try:
                report = UnderwritingReport(**response.report)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "report",
                    "content": f"Report generated for {report.company_name}",
                    "report_data": response.report,
                })
            except ValidationError as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Report validation failed: {e}",
                })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.message or "Report generation failed.",
            })


if __name__ == "__main__":
    main()
