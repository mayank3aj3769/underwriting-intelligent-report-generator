"""LangGraph node functions for the underwriting intelligence pipeline.

Pipeline:  fetch_companies_house → generate_queries → execute_searches
             → summarize_searches → synthesize_report

Every node receives PipelineState and returns a partial state update dict.
"""

import json
import logging
from datetime import datetime, timezone

from openai import AsyncOpenAI

from agents.state import PipelineState
from config import settings
from services.report_formatter import SIC_DESCRIPTIONS, generate_readable_report
from tools.companies_house import CompaniesHouseTool
from tools.search_api import SerpAPIClient

logger = logging.getLogger(__name__)


# =====================================================================
# Node 1: Fetch Companies House data
# =====================================================================

async def fetch_companies_house(state: PipelineState) -> dict:
    """Pull company profile + officers from the Companies House API."""
    ch = CompaniesHouseTool()
    errors: list[str] = []
    company_number = state["company_number"]
    company_name = state["company_name"]

    profile = await ch.get_company_profile(company_number)
    officers = await ch.get_officers(company_number)

    if not profile:
        errors.append(f"Companies House returned no profile for {company_number}")
        return {
            "company_profile_text": "No Companies House data available.",
            "company_metadata": {},
            "errors": errors,
        }

    sic_descs = [
        f"{code} ({SIC_DESCRIPTIONS.get(code, 'unknown')})"
        for code in profile.sic_codes
    ]
    addr_parts = []
    if profile.registered_office_address:
        for k in ("address_line_1", "address_line_2", "locality", "postal_code", "country"):
            v = profile.registered_office_address.get(k)
            if v:
                addr_parts.append(v)

    active_officers = [o for o in officers if not o.resigned_on]
    officer_lines = [f"  - {o.name} ({o.officer_role})" for o in active_officers[:12]]

    profile_text = (
        f"Company Name: {profile.company_name}\n"
        f"Company Number: {profile.company_number}\n"
        f"Status: {profile.company_status or 'unknown'}\n"
        f"Type: {profile.company_type or 'unknown'}\n"
        f"Incorporated: {profile.date_of_creation or 'unknown'}\n"
        f"SIC Codes: {', '.join(sic_descs) if sic_descs else 'none listed'}\n"
        f"Registered Address: {', '.join(addr_parts) if addr_parts else 'not available'}\n"
        f"Has Charges: {profile.has_charges}\n"
        f"Insolvency History: {profile.has_insolvency_history}\n"
        f"Active Officers ({len(active_officers)}):\n" + "\n".join(officer_lines)
    )

    metadata = {
        "company_name": profile.company_name,
        "company_number": profile.company_number,
        "company_status": profile.company_status,
        "date_of_creation": profile.date_of_creation,
        "sic_codes": profile.sic_codes,
        "has_charges": profile.has_charges,
        "has_insolvency_history": profile.has_insolvency_history,
        "officer_count": len(active_officers),
    }

    logger.info(
        "CH data fetched: %s (%s), SIC=%s, officers=%d",
        profile.company_name, profile.company_number,
        profile.sic_codes, len(active_officers),
    )

    return {
        "company_profile_text": profile_text,
        "company_metadata": metadata,
        "errors": errors,
    }


# =====================================================================
# Node 2: LLM generates targeted search queries
# =====================================================================

async def generate_queries(state: PipelineState) -> dict:
    """Use the LLM to produce diverse, context-aware search queries."""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    errors: list[str] = []

    meta = state.get("company_metadata", {})
    sic_codes = meta.get("sic_codes", [])
    sic_labels = [SIC_DESCRIPTIONS.get(c, c) for c in sic_codes]
    name = state["company_name"]

    prompt = (
        f"You are a research analyst investigating **{name}** "
        f"(Companies House #{state['company_number']}) for an underwriting report.\n\n"
        f"Company context:\n"
        f"  Status: {meta.get('company_status', 'unknown')}\n"
        f"  Incorporated: {meta.get('date_of_creation', 'unknown')}\n"
        f"  SIC activities: {', '.join(sic_labels) if sic_labels else 'not listed'}\n"
        f"  Insolvency history: {meta.get('has_insolvency_history', 'unknown')}\n\n"
        "Generate 8–10 targeted Google search queries that will surface:\n"
        "1. What the business does, its products/services, and revenue model\n"
        "2. Key competitors and market position\n"
        "3. Customer reviews and satisfaction (e.g. Trustpilot, G2)\n"
        "4. Recent news, funding rounds, or regulatory actions\n"
        "5. Financial performance indicators\n"
        "6. Industry / sectoral outlook and market trends\n"
        "7. Business risks, controversies, or legal issues\n"
        "8. Key partnerships, clients, or growth signals\n\n"
        "Return a JSON object: {\"queries\": [\"query1\", \"query2\", ...]}\n"
        "Each query should be specific and phrased for Google."
    )

    try:
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate targeted search queries for business "
                        "intelligence research. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        queries: list[str] = data.get("queries", [])
    except Exception as e:
        errors.append(f"LLM query generation failed: {e}")
        queries = []

    logger.info("LLM generated %d search queries", len(queries))
    return {"search_queries": queries, "errors": errors}


# =====================================================================
# Node 3: Execute all searches via SerpAPI
# =====================================================================

async def execute_searches(state: PipelineState) -> dict:
    """Run every LLM-generated query through SerpAPI (web + news)."""
    client = SerpAPIClient()
    all_results: list[dict] = []
    errors: list[str] = []

    for query in state.get("search_queries", []):
        try:
            hits = await client.search(query, num=5)
            for h in hits:
                h["query"] = query
            all_results.extend(hits)
            logger.info("SerpAPI web: '%s' → %d results", query, len(hits))
        except Exception as e:
            errors.append(f"SerpAPI web search failed for '{query}': {e}")
            logger.warning("SerpAPI web failed for '%s': %s", query, e)

    company_name = state["company_name"]
    try:
        news = await client.search_news(f"{company_name} latest news", num=8)
        for n in news:
            n["query"] = f"{company_name} latest news"
        all_results.extend(news)
        logger.info("SerpAPI news: %d results", len(news))
    except Exception as e:
        errors.append(f"SerpAPI news search failed: {e}")

    logger.info("Total search results collected: %d", len(all_results))
    return {"search_results": all_results, "errors": errors}


# =====================================================================
# Node 4: LLM summarises all search results
# =====================================================================

async def summarize_searches(state: PipelineState) -> dict:
    """Produce a structured, citation-backed intelligence briefing."""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    errors: list[str] = []

    results_block = _format_search_results(state.get("search_results", []))
    name = state["company_name"]

    prompt = (
        f"You are a research analyst preparing a deep intelligence briefing "
        f"about **{name}** for an underwriting assessment.\n\n"
        f"Below are search results from multiple queries. "
        f"Synthesise them into a structured briefing.\n\n"
        f"SEARCH RESULTS:\n{results_block}\n\n"
        "Produce a briefing with these sections:\n\n"
        "1. BUSINESS MODEL\n"
        "   a) What does this company actually do? Describe its core operations.\n"
        "   b) How does it make money? Be specific about revenue streams "
        "(subscriptions, fees, commissions, licensing, etc.).\n"
        "   c) Who are its customers? Identify segments (consumers, SMBs, "
        "enterprise, governments, etc.) and geographies.\n"
        "   d) What are its key products/services?\n\n"
        "2. COMPETITIVE LANDSCAPE\n"
        "   a) Name specific competitors and explain WHY each is relevant.\n"
        "   b) Assess the DEGREE of competition (low/medium/high) with "
        "evidence-backed reasoning.\n"
        "   c) Describe the company's market position — leader, challenger, "
        "niche player?\n"
        "   d) Identify competitive advantages AND disadvantages.\n"
        "   e) Explain WHY the competitive intensity matters for assessing "
        "this business's risk profile.\n\n"
        "3. COMPANY QUALITY SIGNALS\n"
        "   a) Customer reviews: Trustpilot ratings, G2, app store reviews — "
        "include specific numbers/ratings if available.\n"
        "   b) Trade press reputation: What do industry publications say?\n"
        "   c) Regulatory standing: Any FCA actions, compliance issues, "
        "awards?\n"
        "   d) For EACH signal, assess its STRENGTH: strong (multiple "
        "corroborating sources), moderate (one credible source), or "
        "weak (anecdotal/unverified).\n"
        "   e) Explicitly note where evidence is THIN, ABSENT, or "
        "CONFLICTING.\n\n"
        "4. RECENT DEVELOPMENTS\n"
        "   Latest news, funding, partnerships, regulatory actions.\n\n"
        "5. BUSINESS OUTLOOK\n"
        "   Growth trajectory, expansion plans, strategic direction.\n\n"
        "6. SECTORAL OUTLOOK\n"
        "   Industry trends, market size, regulatory environment.\n\n"
        "7. RISK INDICATORS\n"
        "   Controversies, legal issues, negative press, financial "
        "concerns.\n\n"
        "RULES:\n"
        "- Cite EVERY factual claim with [Source: <url>]\n"
        "- If a section has no supporting data, state "
        "\"Insufficient data available\" and explain what was searched for\n"
        "- Do NOT infer or speculate beyond what the search results state\n"
        "- When sources conflict, present BOTH sides and note the conflict\n"
        "- Distinguish between well-supported claims and weakly-evidenced ones\n"
    )

    try:
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce grounded, citation-backed business "
                        "intelligence briefings. Every claim must reference "
                        "a specific search result URL."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        summary = resp.choices[0].message.content or ""
    except Exception as e:
        errors.append(f"LLM search summarisation failed: {e}")
        summary = "Search summarisation failed."

    logger.info("Search summary generated (%d chars)", len(summary))
    return {"search_summary": summary, "errors": errors}


# =====================================================================
# Node 5: Final report synthesis
# =====================================================================

async def synthesize_report(state: PipelineState) -> dict:
    """Combine CH filing data + search summary into the final report."""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    errors: list[str] = []

    name = state["company_name"]
    number = state["company_number"]
    meta = state.get("company_metadata", {})

    prompt = (
        f"You are a senior underwriting analyst producing the final "
        f"intelligence report for **{name}** ({number}).\n\n"
        f"COMPANIES HOUSE FILING DATA:\n"
        f"{state.get('company_profile_text', 'Not available')}\n\n"
        f"RESEARCH INTELLIGENCE BRIEFING:\n"
        f"{state.get('search_summary', 'Not available')}\n\n"
        "Produce a JSON report with this EXACT structure:\n"
        "{\n"
        '  "business_model": {\n'
        '    "description": "comprehensive explanation of what this company '
        'actually does — grounded in evidence, not generic",\n'
        '    "revenue_model": "how it generates revenue — be specific '
        '(subscriptions, fees, commissions, etc.)",\n'
        '    "key_products_services": ["product1", "product2"],\n'
        '    "customer_segments": ["who buys from them — be specific"],\n'
        '    "geographies": ["key markets/regions served"],\n'
        '    "evidence_basis": "which sources informed this section and '
        'how reliable they are"\n'
        "  },\n"
        '  "competitive_landscape": {\n'
        '    "industry": "industry name",\n'
        '    "market_position": "where this company sits — leader, '
        'challenger, niche player, etc., with reasoning",\n'
        '    "competitors": [\n'
        '      {"name": "X", "description": "what they do", '
        '"relevance": "why this competitor matters for comparison"}\n'
        "    ],\n"
        '    "competition_degree": "low | medium | high",\n'
        '    "competitive_advantages": ["specific edges this company has"],\n'
        '    "competitive_disadvantages": ["specific weaknesses vs peers"],\n'
        '    "reasoning": "a structured analysis of WHY competition is at '
        'this level and what it means for underwriting this business — '
        'not just a label but an argument backed by evidence",\n'
        '    "evidence_basis": "which sources informed this section and '
        'how reliable they are"\n'
        "  },\n"
        '  "quality_signals": [\n'
        "    {\n"
        '      "signal": "the observed quality indicator",\n'
        '      "sentiment": "positive | negative | neutral",\n'
        '      "strength": "strong | moderate | weak — based on source '
        'reliability and corroboration",\n'
        '      "source": "source name (e.g. Trustpilot, FCA, Reuters)",\n'
        '      "url": "url or null",\n'
        '      "detail": "supporting quote, rating, or data point"\n'
        "    }\n"
        "  ],\n"
        '  "signal_coverage_assessment": "overall assessment of evidence '
        "quality — is there enough data to be confident about this "
        "company's quality? Which categories (reviews, press, regulatory) "
        'are well-covered vs thin?",\n'
        '  "data_gaps": ["specific areas where evidence is thin or absent"],\n'
        '  "conflicting_signals": ["describe any contradictions between '
        'sources"],\n'
        '  "business_outlook": "evidence-backed assessment of business '
        'trajectory",\n'
        '  "sectoral_outlook": "evidence-backed assessment of industry '
        'trajectory",\n'
        '  "uncertainty_flags": {\n'
        '    "missing_data": ["what we could not find despite searching"],\n'
        '    "conflicting_evidence": ["specific conflicts between sources"],\n'
        '    "low_confidence_areas": ["areas where conclusions are uncertain '
        'and why"]\n'
        "  }\n"
        "}\n\n"
        "RULES:\n"
        "- Every claim MUST be grounded in the data above — no fabrication\n"
        "- Each section must be EXPLAINABLE: a reader should understand "
        "WHY you reached each conclusion\n"
        "- Surface uncertainty rather than papering over it — if evidence "
        "is thin, say so explicitly\n"
        "- When sources conflict, present both sides\n"
        "- Quality signals: for each signal, assess its strength (strong = "
        "multiple corroborating sources, moderate = one credible source, "
        "weak = anecdotal/unverified)\n"
        "- Competitive landscape: go BEYOND listing competitors — explain "
        "the degree of competition and why that matters for this business\n"
        "- Business model: be specific about what the company does, how it "
        "makes money, and who its customers are — avoid generic descriptions\n"
    )

    try:
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior underwriting analyst. "
                        "Produce grounded intelligence reports in strict JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        analysis = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        errors.append(f"LLM report synthesis failed: {e}")
        return {"final_report": {}, "errors": errors}

    report = _build_report(state, analysis, meta, errors)
    logger.info("Final report synthesised for %s", name)
    return {"final_report": report, "errors": errors}


# =====================================================================
# Helpers
# =====================================================================

def _format_search_results(results: list[dict]) -> str:
    """Format raw SerpAPI results into a numbered block for the LLM."""
    if not results:
        return "(no search results available)"
    lines: list[str] = []
    for i, r in enumerate(results, 1):
        rtype = r.get("type", "web")
        date_str = f" | {r['date']}" if r.get("date") else ""
        lines.append(
            f"[{i}] ({rtype}{date_str}) {r.get('title', '')}\n"
            f"    URL: {r.get('url', 'N/A')}\n"
            f"    Snippet: {r.get('snippet', '')}\n"
            f"    Query: {r.get('query', '')}"
        )
    return "\n".join(lines)


def _build_report(
    state: PipelineState,
    analysis: dict,
    meta: dict,
    node_errors: list[str],
) -> dict:
    """Construct the final report dict (serialisable as UnderwritingReport)."""
    from schemas.report import (
        BusinessModelSummary,
        Citation,
        CompanyQualitySignals,
        CompetitionDegree,
        CompetitiveLandscape,
        Competitor,
        ConfidenceLevel,
        QualitySignal,
        SignalStrength,
        UncertaintyFlags,
        UnderwritingReport,
    )

    citations = _extract_citations(state.get("search_results", []))

    bm = analysis.get("business_model", {})
    cl = analysis.get("competitive_landscape", {})
    qs_raw = analysis.get("quality_signals", [])
    uf = analysis.get("uncertainty_flags", {})

    competition = cl.get("competition_degree", "medium").lower()
    if competition not in ("low", "medium", "high"):
        competition = "medium"

    def _parse_strength(val: str) -> SignalStrength:
        val = (val or "moderate").lower()
        if val in ("strong", "moderate", "weak"):
            return SignalStrength(val)
        return SignalStrength.MODERATE

    signals = [
        QualitySignal(
            signal=s.get("signal", ""),
            sentiment=s.get("sentiment", "neutral"),
            strength=_parse_strength(s.get("strength")),
            source=s.get("source", "unknown"),
            url=s.get("url"),
            detail=s.get("detail"),
        )
        for s in qs_raw
    ]
    pos = sum(1 for s in signals if s.sentiment == "positive")
    neg = sum(1 for s in signals if s.sentiment == "negative")

    source_types = set()
    source_types.add("companies_house")
    if state.get("search_results"):
        for r in state["search_results"]:
            source_types.add(r.get("type", "web"))

    evidence_count = len(state.get("search_results", []))
    if state.get("company_profile_text"):
        evidence_count += 1

    confidence = ConfidenceLevel.HIGH if evidence_count > 15 else (
        ConfidenceLevel.MEDIUM if evidence_count > 5 else ConfidenceLevel.LOW
    )

    report = UnderwritingReport(
        company_name=state["company_name"],
        company_number=state["company_number"],
        report_generated_at=datetime.now(timezone.utc).isoformat(),
        sources_used=sorted(source_types),
        business_model=BusinessModelSummary(
            description=bm.get("description", "Unable to determine"),
            revenue_model=bm.get("revenue_model"),
            key_products_services=bm.get("key_products_services", []),
            customer_segments=bm.get("customer_segments", []),
            geographies=bm.get("geographies", []),
            evidence_basis=bm.get("evidence_basis"),
            citations=citations,
        ),
        competitive_landscape=CompetitiveLandscape(
            industry=cl.get("industry", "Unknown"),
            sic_codes=meta.get("sic_codes", []),
            market_position=cl.get("market_position"),
            competitors=[
                Competitor(
                    name=c.get("name", ""),
                    description=c.get("description"),
                    relevance=c.get("relevance"),
                )
                for c in cl.get("competitors", [])
            ],
            competition_degree=CompetitionDegree(competition),
            competitive_advantages=cl.get("competitive_advantages", []),
            competitive_disadvantages=cl.get("competitive_disadvantages", []),
            reasoning=cl.get("reasoning", "Insufficient data"),
            evidence_basis=cl.get("evidence_basis"),
            citations=citations,
        ),
        quality_signals=CompanyQualitySignals(
            signals=signals,
            positive_count=pos,
            negative_count=neg,
            confidence=confidence,
            signal_coverage_assessment=analysis.get("signal_coverage_assessment"),
            data_gaps=analysis.get("data_gaps", []),
            conflicting_signals=analysis.get("conflicting_signals", []),
            missing_data=uf.get("missing_data", []),
            citations=citations,
        ),
        uncertainty_flags=UncertaintyFlags(
            missing_data=uf.get("missing_data", []),
            conflicting_evidence=uf.get("conflicting_evidence", []),
            low_confidence_areas=uf.get("low_confidence_areas", []),
        ),
        business_outlook=analysis.get("business_outlook"),
        sectoral_outlook=analysis.get("sectoral_outlook"),
        raw_evidence_count=evidence_count,
    )

    report.readable_report = generate_readable_report(report)
    return report.model_dump()


def _extract_citations(results: list[dict]) -> list:
    """Deduplicated citation list from raw search results."""
    from schemas.report import Citation

    seen: set[str] = set()
    citations: list[Citation] = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            citations.append(Citation(
                source=r.get("type", "web"),
                url=url,
                detail=r.get("title", ""),
            ))
    return citations
