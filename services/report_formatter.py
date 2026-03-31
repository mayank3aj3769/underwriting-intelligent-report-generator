"""SIC code descriptions and readable report formatting."""

from schemas.report import UnderwritingReport

SIC_DESCRIPTIONS: dict[str, str] = {
    "62011": "Computer programming activities",
    "62012": "Business and domestic software development",
    "62020": "Information technology consultancy activities",
    "62090": "Other information technology service activities",
    "64110": "Central banking",
    "64191": "Banks",
    "64192": "Building societies",
    "64205": "Activities of financial services holding companies",
    "64209": "Activities of other holding companies n.e.c.",
    "64301": "Activities of investment trusts",
    "64999": "Financial intermediation not elsewhere classified",
    "66110": "Administration of financial markets",
    "66190": "Other activities auxiliary to financial services",
    "66220": "Activities of insurance agents and brokers",
    "70100": "Activities of head offices",
    "70210": "Public relations and communication activities",
    "70229": "Management consultancy activities other than financial management",
    "73110": "Advertising agencies",
    "73120": "Media representation services",
    "82110": "Combined office administrative service activities",
    "82990": "Other business support service activities n.e.c.",
    "47110": "Retail sale in non-specialised stores",
    "47910": "Retail sale via mail order houses or via Internet",
    "56101": "Licensed restaurants",
    "56102": "Unlicensed restaurants and cafes",
    "58110": "Book publishing",
    "58290": "Other software publishing",
    "86101": "Hospital activities",
    "86210": "General medical practice activities",
    "96090": "Other service activities n.e.c.",
}


def generate_readable_report(report: UnderwritingReport) -> str:
    """Render the structured report as human-readable Markdown."""
    lines: list[str] = []

    lines.append(f"# Underwriting Intelligence Report: {report.company_name}")
    lines.append(f"**Company Number:** {report.company_number}")
    lines.append(f"**Generated:** {report.report_generated_at}")
    lines.append(f"**Sources Used:** {', '.join(report.sources_used)}")
    lines.append(f"**Evidence Items:** {report.raw_evidence_count}")
    lines.append("")

    # ── A: Business Model ──
    lines.append("## A. Business Model Summary")
    lines.append(report.business_model.description)
    if report.business_model.revenue_model:
        lines.append(f"\n**Revenue Model:** {report.business_model.revenue_model}")
    if report.business_model.key_products_services:
        lines.append(
            f"**Key Products/Services:** "
            f"{', '.join(report.business_model.key_products_services)}"
        )
    if report.business_model.customer_segments:
        lines.append(
            f"**Customer Segments:** "
            f"{', '.join(report.business_model.customer_segments)}"
        )
    if report.business_model.geographies:
        lines.append(
            f"**Geographies:** {', '.join(report.business_model.geographies)}"
        )
    if report.business_model.evidence_basis:
        lines.append(
            f"\n*Evidence basis: {report.business_model.evidence_basis}*"
        )
    lines.append("")

    # ── B: Competitive Landscape ──
    lines.append("## B. Competitive Landscape")
    lines.append(f"**Industry:** {report.competitive_landscape.industry}")
    if report.competitive_landscape.sic_codes:
        lines.append(
            f"**SIC Codes:** {', '.join(report.competitive_landscape.sic_codes)}"
        )
    lines.append(
        f"**Competition Degree:** "
        f"{report.competitive_landscape.competition_degree.value}"
    )
    if report.competitive_landscape.market_position:
        lines.append(
            f"**Market Position:** {report.competitive_landscape.market_position}"
        )
    lines.append(f"\n**Analysis:** {report.competitive_landscape.reasoning}")
    if report.competitive_landscape.competitors:
        lines.append("\n**Key Competitors:**")
        for comp in report.competitive_landscape.competitors:
            desc = f" -- {comp.description}" if comp.description else ""
            rel = f" _{comp.relevance}_" if comp.relevance else ""
            lines.append(f"- **{comp.name}**{desc}{rel}")
    if report.competitive_landscape.competitive_advantages:
        lines.append("\n**Competitive Advantages:**")
        for adv in report.competitive_landscape.competitive_advantages:
            lines.append(f"  + {adv}")
    if report.competitive_landscape.competitive_disadvantages:
        lines.append("\n**Competitive Disadvantages:**")
        for dis in report.competitive_landscape.competitive_disadvantages:
            lines.append(f"  - {dis}")
    if report.competitive_landscape.evidence_basis:
        lines.append(
            f"\n*Evidence basis: {report.competitive_landscape.evidence_basis}*"
        )
    lines.append("")

    # ── C: Quality Signals ──
    lines.append("## C. Company Quality Signals")
    lines.append(f"**Confidence:** {report.quality_signals.confidence.value}")
    lines.append(f"**Positive Signals:** {report.quality_signals.positive_count}")
    lines.append(f"**Negative Signals:** {report.quality_signals.negative_count}")
    if report.quality_signals.signal_coverage_assessment:
        lines.append(
            f"\n**Signal Coverage:** {report.quality_signals.signal_coverage_assessment}"
        )
    if report.quality_signals.signals:
        lines.append("\n**Signals:**")
        for sig in report.quality_signals.signals:
            icon = {"positive": "+", "negative": "-", "neutral": "~"}.get(
                sig.sentiment, "?"
            )
            strength_tag = f" [{sig.strength.value}]" if sig.strength else ""
            src = f" [{sig.url}]" if sig.url else ""
            lines.append(
                f"  [{icon}]{strength_tag} {sig.signal} (source: {sig.source}{src})"
            )
            if sig.detail:
                lines.append(f"      Detail: {sig.detail}")
    if report.quality_signals.data_gaps:
        lines.append("\n**Data Gaps:**")
        for gap in report.quality_signals.data_gaps:
            lines.append(f"  ! {gap}")
    if report.quality_signals.conflicting_signals:
        lines.append("\n**Conflicting Signals:**")
        for conflict in report.quality_signals.conflicting_signals:
            lines.append(f"  ? {conflict}")
    if report.quality_signals.missing_data:
        lines.append("\n**Missing Data:**")
        for md in report.quality_signals.missing_data:
            lines.append(f"  - {md}")
    lines.append("")

    # ── D: Business Outlook ──
    if report.business_outlook:
        lines.append("## D. Business Outlook")
        lines.append(report.business_outlook)
        lines.append("")

    # ── E: Sectoral Outlook ──
    if report.sectoral_outlook:
        lines.append("## E. Sectoral Outlook")
        lines.append(report.sectoral_outlook)
        lines.append("")

    # ── F: Uncertainty ──
    lines.append("## F. Uncertainty Flags")
    if report.uncertainty_flags.missing_data:
        lines.append("**Missing Data:**")
        for item in report.uncertainty_flags.missing_data:
            lines.append(f"  - {item}")
    if report.uncertainty_flags.conflicting_evidence:
        lines.append("**Conflicting Evidence:**")
        for item in report.uncertainty_flags.conflicting_evidence:
            lines.append(f"  - {item}")
    if report.uncertainty_flags.low_confidence_areas:
        lines.append("**Low-Confidence Areas:**")
        for item in report.uncertainty_flags.low_confidence_areas:
            lines.append(f"  - {item}")
    if not any([
        report.uncertainty_flags.missing_data,
        report.uncertainty_flags.conflicting_evidence,
        report.uncertainty_flags.low_confidence_areas,
    ]):
        lines.append("No significant uncertainty flags.")
    lines.append("")

    # ── G: Citations ──
    lines.append("## G. Sources & Citations")
    for cit in report.business_model.citations[:25]:
        url_part = f" ({cit.url})" if cit.url else ""
        lines.append(f"- [{cit.source}] {cit.detail or ''}{url_part}")

    lines.append("\n---")
    lines.append(
        "*Report generated by Underwriting Intelligence System (LangGraph pipeline)*"
    )
    return "\n".join(lines)
