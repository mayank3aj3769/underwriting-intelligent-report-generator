"""LangGraph node functions for the underwriting intelligence pipeline.

Pipeline:  fetch_companies_house → generate_queries → execute_searches
             → summarize_searches → evaluate_sufficiency
             → (if insufficient) generate_gap_queries → execute_searches
               → summarize_searches → evaluate_sufficiency
             → synthesize_report

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

MAX_SUFFICIENCY_ITERATIONS = 2


# =====================================================================
# Node 1: Fetch Companies House data
# =====================================================================

async def fetch_companies_house(state: PipelineState) -> dict:
    """Pull company profile + officers from the Companies House API."""
    ch = CompaniesHouseTool()
    errors: list[str] = []
    company_number = state["company_number"]

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
        "Generate exactly 6 targeted Google search queries that will surface:\n"
        "1. Business model, products/services, and revenue model\n"
        "2. Revenue figures: current revenue, annual turnover, financial results, "
        "revenue growth — use terms like 'revenue', 'turnover', 'financial results', "
        "'annual report' in the query\n"
        "3. Key competitors and market position\n"
        "4. Customer reviews and reputation (Trustpilot, G2, press)\n"
        "5. Recent news, funding, or regulatory actions\n"
        "6. Industry outlook and sectoral trends\n\n"
        "Return a JSON object: {\"queries\": [\"query1\", ..., \"query6\"]}\n"
        "IMPORTANT: Query #2 MUST specifically target financial/revenue data. "
        "Example: '\"COMPANY NAME\" revenue turnover financial results 2024 2025'.\n"
        "Each query should be specific, phrased for Google, and maximise coverage across these 6 areas."
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
        queries: list[str] = data.get("queries", [])[:6]
    except Exception as e:
        errors.append(f"LLM query generation failed: {e}")
        queries = []

    logger.info("LLM generated %d search queries", len(queries))
    return {"search_queries": queries, "errors": errors}


# =====================================================================
# Node 3: Execute all searches via SerpAPI
# =====================================================================

async def execute_searches(state: PipelineState) -> dict:
    """Run all LLM-generated queries through SerpAPI in parallel.

    On iteration > 0, merges new results with existing ones (deduped by URL).
    """
    import asyncio

    client = SerpAPIClient()
    errors: list[str] = []
    queries = state.get("search_queries", [])
    company_name = state["company_name"]
    iteration = state.get("iteration_count", 0)

    async def _web_search(query: str) -> tuple[str, list[dict] | Exception]:
        try:
            hits = await client.search(query, num=5)
            for h in hits:
                h["query"] = query
            return query, hits
        except Exception as e:
            return query, e

    async def _news_search() -> tuple[str, list[dict] | Exception]:
        q = f"{company_name} latest news"
        try:
            hits = await client.search_news(q, num=8)
            for h in hits:
                h["query"] = q
            return q, hits
        except Exception as e:
            return q, e

    tasks = [_web_search(q) for q in queries]
    if iteration == 0:
        tasks.append(_news_search())

    results = await asyncio.gather(*tasks)

    new_results: list[dict] = []
    for query, outcome in results:
        if isinstance(outcome, Exception):
            errors.append(f"SerpAPI search failed for '{query}': {outcome}")
            logger.warning("SerpAPI failed for '%s': %s", query, outcome)
        else:
            new_results.extend(outcome)
            logger.info("SerpAPI: '%s' -> %d results", query, len(outcome))

    existing_results = state.get("search_results", []) if iteration > 0 else []
    seen_urls = {r.get("url") for r in existing_results if r.get("url")}
    for r in new_results:
        if r.get("url") not in seen_urls:
            existing_results.append(r)
            seen_urls.add(r.get("url"))

    logger.info(
        "Search iteration %d: %d new results, %d total (from %d queries)",
        iteration, len(new_results), len(existing_results), len(tasks),
    )
    return {"search_results": existing_results, "errors": errors}


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
        "1. BUSINESS MODEL & REVENUE\n"
        "   a) What does this company actually do? Describe its core operations.\n"
        "   b) How does it make money? Be specific about revenue streams "
        "(subscriptions, fees, commissions, licensing, etc.).\n"
        "   c) CURRENT REVENUE: What is the most recent publicly available "
        "revenue or turnover figure? Include the year/period and currency. "
        "If the company is private and revenue is not publicly disclosed, "
        "state that explicitly.\n"
        "   d) REVENUE TREND: Is revenue growing or declining? By how much? "
        "Over what period? If trend data is not available, state that explicitly.\n"
        "   e) Who are its customers? Identify segments (consumers, SMBs, "
        "enterprise, governments, etc.) and geographies.\n"
        "   f) What are its key products/services?\n\n"
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
# Node 5: Evaluate evidence sufficiency
# =====================================================================

async def evaluate_sufficiency(state: PipelineState) -> dict:
    """Analyse collected evidence and decide whether it is sufficient.

    Checks:
      - Total source count
      - Coverage across key sections (business, competition, quality, news)
      - Presence of competitor mentions
      - Presence of review / news content
    Returns evidence_metrics, updated iteration_count, and sufficiency_flag.
    """
    results = state.get("search_results", [])
    summary = state.get("search_summary", "")
    iteration = state.get("iteration_count", 0)
    summary_lower = summary.lower()

    total_sources = len(results)

    result_types = {r.get("type", "web") for r in results}
    result_snippets = " ".join(
        (r.get("snippet", "") + " " + r.get("title", "")).lower()
        for r in results
    )

    has_business_info = any(
        kw in result_snippets
        for kw in ("business model", "product", "service", "customer")
    )
    has_revenue_data = any(
        kw in result_snippets
        for kw in (
            "revenue", "turnover", "annual report", "financial results",
            "profit", "£", "$", "million", "billion", "mn", "bn",
        )
    )
    has_competitors = any(
        kw in result_snippets or kw in summary_lower
        for kw in ("competitor", "competes with", "rival", "market share", "vs ")
    )
    has_reviews = any(
        kw in result_snippets
        for kw in ("trustpilot", "review", "rating", "g2.com", "glassdoor")
    )
    has_news = "news" in result_types or any(
        kw in result_snippets for kw in ("news", "announced", "funding", "raised")
    )
    has_financial_signals = any(
        kw in result_snippets
        for kw in ("fca", "insolvency", "charge", "regulatory", "fine", "compliance")
    )

    section_coverage = {
        "business_model": has_business_info,
        "revenue": has_revenue_data,
        "competition": has_competitors,
        "quality_signals": has_reviews,
        "news": has_news,
        "financial_regulatory": has_financial_signals,
    }

    covered_count = sum(1 for v in section_coverage.values() if v)
    total_sections = len(section_coverage)

    source_score = min(total_sources / 20.0, 1.0) * 0.3
    coverage_score = (covered_count / total_sections) * 0.5
    summary_score = min(len(summary) / 3000.0, 1.0) * 0.2
    confidence_score = round(source_score + coverage_score + summary_score, 2)

    missing_sections: list[str] = [
        section for section, covered in section_coverage.items() if not covered
    ]

    is_sufficient = (
        confidence_score >= 0.6
        and total_sources >= 8
        and covered_count >= 3
    )

    if iteration >= MAX_SUFFICIENCY_ITERATIONS:
        reasoning = (
            f"Max iterations ({MAX_SUFFICIENCY_ITERATIONS}) reached. "
            f"Proceeding with available evidence "
            f"(confidence={confidence_score}, sources={total_sources}, "
            f"covered={covered_count}/{total_sections})."
        )
        is_sufficient = True
        logger.info("Sufficiency: max iterations reached, forcing proceed. %s", reasoning)
    elif is_sufficient:
        reasoning = (
            f"Evidence sufficient at iteration {iteration}: "
            f"confidence={confidence_score}, sources={total_sources}, "
            f"covered={covered_count}/{total_sections}."
        )
        logger.info("Sufficiency: PASS. %s", reasoning)
    else:
        reasoning = (
            f"Evidence insufficient at iteration {iteration}: "
            f"confidence={confidence_score}, sources={total_sources}, "
            f"covered={covered_count}/{total_sections}. "
            f"Missing: {', '.join(missing_sections)}."
        )
        logger.info("Sufficiency: FAIL — triggering additional data gathering. %s", reasoning)

    metrics = {
        "total_sources": total_sources,
        "has_business_info": has_business_info,
        "has_revenue_data": has_revenue_data,
        "has_competitors": has_competitors,
        "has_reviews": has_reviews,
        "has_news": has_news,
        "has_financial_signals": has_financial_signals,
        "section_coverage": section_coverage,
        "confidence_score": confidence_score,
        "missing_sections": missing_sections,
        "is_sufficient": is_sufficient,
        "reasoning": reasoning,
    }

    return {
        "evidence_metrics": metrics,
        "iteration_count": iteration + 1,
        "sufficiency_flag": is_sufficient,
    }


# =====================================================================
# Node 6: Generate gap-filling queries for missing sections
# =====================================================================

_GAP_QUERY_TEMPLATES: dict[str, str] = {
    "business_model": '"{name}" business model products services',
    "revenue": '"{name}" revenue turnover financial results annual report',
    "competition": '"{name}" competitors market share industry rivals',
    "quality_signals": '"{name}" Trustpilot reviews customer ratings reputation',
    "news": '"{name}" latest news announcements funding',
    "financial_regulatory": '"{name}" FCA regulation compliance financial conduct',
}


async def generate_gap_queries(state: PipelineState) -> dict:
    """Generate targeted queries that fill gaps identified by evaluate_sufficiency."""
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    errors: list[str] = []

    metrics = state.get("evidence_metrics", {})
    missing = metrics.get("missing_sections", [])
    name = state["company_name"]
    meta = state.get("company_metadata", {})
    sic_codes = meta.get("sic_codes", [])
    sic_labels = [SIC_DESCRIPTIONS.get(c, c) for c in sic_codes]

    if not missing:
        logger.info("No gaps to fill — no additional queries generated")
        return {"search_queries": [], "errors": errors}

    gap_descriptions = {
        "business_model": "business model, products/services, and how the company operates",
        "revenue": "current revenue, annual turnover, financial results, and revenue growth trend",
        "competition": "competitors, market position, and competitive landscape",
        "quality_signals": "customer reviews (Trustpilot, G2), reputation signals, and ratings",
        "news": "recent news, funding rounds, partnerships, or regulatory actions",
        "financial_regulatory": "FCA status, regulatory compliance, financial conduct issues",
    }

    gap_list = "\n".join(
        f"- {gap_descriptions.get(s, s)}" for s in missing
    )

    prompt = (
        f"You are a research analyst investigating **{name}** "
        f"(Companies House #{state['company_number']}).\n\n"
        f"SIC activities: {', '.join(sic_labels) if sic_labels else 'not listed'}\n\n"
        f"Previous research found INSUFFICIENT evidence in these areas:\n{gap_list}\n\n"
        f"Generate {min(len(missing) * 2, 5)} highly targeted Google search queries "
        f"that will SPECIFICALLY fill these gaps. Focus ONLY on the missing areas.\n\n"
        "Return a JSON object: {\"queries\": [\"query1\", ...]}\n"
        "Each query should be specific, phrased for Google, and designed to find "
        "the exact type of information that is missing."
    )

    try:
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate targeted gap-filling search queries for business "
                        "intelligence research. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        queries: list[str] = data.get("queries", [])[:5]
    except Exception as e:
        logger.warning("LLM gap query generation failed, using templates: %s", e)
        errors.append(f"LLM gap query generation failed (using templates): {e}")
        queries = [
            _GAP_QUERY_TEMPLATES[s].format(name=name)
            for s in missing
            if s in _GAP_QUERY_TEMPLATES
        ][:5]

    logger.info(
        "Generated %d gap-filling queries for missing sections: %s",
        len(queries), missing,
    )
    return {"search_queries": queries, "errors": errors}


# =====================================================================
# Node 7: Final report synthesis
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
        '    "current_revenue": "most recent publicly available revenue/turnover '
        'figure with year and currency (e.g. \'£1.1bn revenue in FY2024\'). '
        'If the company is private and revenue is not publicly disclosed, '
        'state: \'Revenue figures are not publicly available — [company] is '
        'a private company that does not disclose financial results.\'",\n'
        '    "revenue_trend": "revenue growth or decline trajectory with '
        'specific numbers and time period (e.g. \'Revenue grew 35% YoY from '
        '£800m in 2023 to £1.1bn in 2024\'). If trend data is not available, '
        'state that explicitly and explain why.",\n'
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
# Routing function for conditional edge
# =====================================================================

def sufficiency_router(state: PipelineState) -> str:
    """Decide whether to loop for more evidence or proceed to synthesis."""
    if state.get("sufficiency_flag", False):
        return "synthesize_report"
    return "generate_gap_queries"


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

    metrics = state.get("evidence_metrics", {})

    report = UnderwritingReport(
        company_name=state["company_name"],
        company_number=state["company_number"],
        report_generated_at=datetime.now(timezone.utc).isoformat(),
        sources_used=sorted(source_types),
        business_model=BusinessModelSummary(
            description=bm.get("description", "Unable to determine"),
            revenue_model=bm.get("revenue_model"),
            current_revenue=bm.get("current_revenue"),
            revenue_trend=bm.get("revenue_trend"),
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
        evidence_confidence_score=metrics.get("confidence_score", 0.0),
        evidence_iterations=state.get("iteration_count", 1),
        evidence_gaps_found=metrics.get("missing_sections", []),
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
