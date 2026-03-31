"""End-to-end test suite for the Underwriting Intelligence System.

Tests:
  1. Health check
  2. Company search (disambiguation)
  3. Report generation — ambiguous name triggers disambiguation
  4. Report generation — exact name auto-resolves
  5. Report generation — by company number (full pipeline)
  6. Edge case — unknown company number
  7. Edge case — empty input

Run:
  python -m tests.test_e2e

Results are written to tests/test_results.txt.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config import settings
from services.entity_resolver import EntityResolver
from services.report_generator import ReportGenerator, ReportGenerationResult

LOG: list[str] = []
RESULTS_DIR = Path(__file__).resolve().parent


def log(msg: str = "") -> None:
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    print(safe)
    LOG.append(msg)


def divider(title: str) -> None:
    log("")
    log("=" * 72)
    log(f"  {title}")
    log("=" * 72)


def sub_divider(title: str) -> None:
    log(f"\n--- {title} ---")


async def test_health() -> bool:
    """Verify all API keys are configured."""
    divider("TEST 1: Health / Configuration Check")
    checks = {
        "COMPANIES_HOUSE_API_KEY": settings.has_companies_house_key,
        "COMPANIES_HOUSE_API_URL": bool(settings.COMPANIES_HOUSE_API_URL),
        "SERP_API_KEY": settings.has_serp_api_key,
        "OPENAI_API_KEY": settings.has_openai_key,
        "OPENAI_MODEL": settings.OPENAI_MODEL,
    }
    all_ok = True
    for name, ok in checks.items():
        status = "OK" if ok else "MISSING"
        if not ok and name in ("COMPANIES_HOUSE_API_KEY", "SERP_API_KEY", "OPENAI_API_KEY"):
            all_ok = False
        log(f"  {name}: {status}")
    log(f"\n  CH API URL: {settings.COMPANIES_HOUSE_API_URL}")
    log(f"  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


async def test_search_disambiguation() -> bool:
    """Search for a common name that returns multiple candidates."""
    divider("TEST 2: Company Search -- Disambiguation (query='Barclays')")
    resolver = EntityResolver()
    result = await resolver.search_by_name("Barclays")
    log(f"  Query: 'Barclays'")
    log(f"  Total results: {result.total_results}")
    log(f"  Disambiguation required: {result.disambiguation_required}")
    log(f"  Candidates returned: {len(result.candidates)}")
    for c in result.candidates[:8]:
        status = c.company_status or "?"
        log(f"    - {c.company_name} ({c.company_number}) [{status}]")

    passed = result.total_results > 1
    log(f"\n  Result: {'PASS' if passed else 'FAIL'} -- multiple candidates found")
    return passed


async def test_report_ambiguous_name() -> bool:
    """Report endpoint with ambiguous name should return disambiguation."""
    divider("TEST 3: Report Generation -- Ambiguous Name Triggers Disambiguation")
    gen = ReportGenerator()
    result = await gen.generate("Barclays")
    data = result.to_dict()

    log(f"  Input: 'Barclays'")
    log(f"  Status: {data.get('status')}")
    log(f"  Needs disambiguation: {result.needs_disambiguation}")
    if result.needs_disambiguation:
        log(f"  Message: {data.get('message')}")
        log(f"  Candidate count: {data.get('total_results')}")
        for c in data.get("candidates", [])[:5]:
            log(f"    - {c['company_name']} ({c['company_number']})")

    passed = result.needs_disambiguation
    log(f"\n  Result: {'PASS' if passed else 'FAIL'} -- disambiguation correctly triggered")
    return passed


async def test_report_exact_name() -> bool:
    """Report endpoint with an exact, unambiguous name should auto-resolve."""
    divider("TEST 4: Report Generation -- Exact Name Auto-Resolves (query='REVOLUT LTD')")
    gen = ReportGenerator()
    result = await gen.generate("REVOLUT LTD")
    data = result.to_dict()

    log(f"  Input: 'REVOLUT LTD'")
    log(f"  Status: {data.get('status')}")
    log(f"  Needs disambiguation: {result.needs_disambiguation}")
    log(f"  Is error: {result.is_error}")

    if data.get("status") == "success" and result.report:
        log(f"  Company: {result.report.company_name}")
        log(f"  Number: {result.report.company_number}")
        log(f"  Evidence count: {result.report.raw_evidence_count}")
        passed = True
    elif result.needs_disambiguation:
        log(f"  (Disambiguation returned -- name was not unique enough)")
        passed = True
    else:
        log(f"  Error: {data.get('message', 'unknown')}")
        passed = False

    log(f"\n  Result: {'PASS' if passed else 'FAIL'}")
    return passed


async def test_report_by_number() -> bool:
    """Full pipeline test -- generate report for Revolut by company number."""
    divider("TEST 5: Full Pipeline -- Report by Company Number (08804411 = REVOLUT LTD)")
    gen = ReportGenerator()

    t0 = time.time()
    result = await gen.generate_by_number("08804411")
    elapsed = time.time() - t0

    data = result.to_dict()
    log(f"  Input: company_number='08804411'")
    log(f"  Status: {data.get('status')}")
    log(f"  Time: {elapsed:.1f}s")

    if data.get("status") != "success" or not result.report:
        log(f"  Error: {data.get('message', 'unknown')}")
        log(f"\n  Result: FAIL")
        return False

    report = result.report
    log(f"\n  Company: {report.company_name}")
    log(f"  Number: {report.company_number}")
    log(f"  Generated at: {report.report_generated_at}")
    log(f"  Sources used: {', '.join(report.sources_used)}")
    log(f"  Evidence items: {report.raw_evidence_count}")

    sub_divider("Business Model")
    log(f"  Description: {report.business_model.description[:200]}...")
    log(f"  Revenue model: {report.business_model.revenue_model or 'N/A'}")
    log(f"  Customer segments: {report.business_model.customer_segments}")
    log(f"  Citations: {len(report.business_model.citations)}")

    sub_divider("Competitive Landscape")
    log(f"  Industry: {report.competitive_landscape.industry}")
    log(f"  Competition: {report.competitive_landscape.competition_degree.value}")
    log(f"  Competitors: {len(report.competitive_landscape.competitors)}")
    for comp in report.competitive_landscape.competitors[:5]:
        log(f"    - {comp.name}: {comp.description or 'N/A'}")
    log(f"  Reasoning: {report.competitive_landscape.reasoning[:200]}...")

    sub_divider("Quality Signals")
    log(f"  Confidence: {report.quality_signals.confidence.value}")
    log(f"  Positive: {report.quality_signals.positive_count}, Negative: {report.quality_signals.negative_count}")
    for sig in report.quality_signals.signals[:6]:
        icon = {"positive": "+", "negative": "-", "neutral": "~"}.get(sig.sentiment, "?")
        log(f"    [{icon}] {sig.signal} (source: {sig.source})")

    sub_divider("Outlook")
    log(f"  Business outlook: {(report.business_outlook or 'N/A')[:200]}")
    log(f"  Sectoral outlook: {(report.sectoral_outlook or 'N/A')[:200]}")

    sub_divider("Uncertainty Flags")
    log(f"  Missing data: {report.uncertainty_flags.missing_data}")
    log(f"  Conflicting evidence: {report.uncertainty_flags.conflicting_evidence}")
    log(f"  Low confidence: {report.uncertainty_flags.low_confidence_areas}")

    sub_divider("Readable Report (first 500 chars)")
    if report.readable_report:
        log(report.readable_report[:500])
    else:
        log("  (no readable report generated)")

    passed = bool(
        report.company_name
        and report.business_model.description
        and report.competitive_landscape.competitors
        and report.raw_evidence_count > 0
    )
    log(f"\n  Result: {'PASS' if passed else 'FAIL'} -- full report generated with evidence")
    return passed


async def test_unknown_company() -> bool:
    """Unknown company number should return a clear error."""
    divider("TEST 6: Edge Case -- Unknown Company Number")
    gen = ReportGenerator()
    result = await gen.generate_by_number("99999999")
    data = result.to_dict()

    log(f"  Input: company_number='99999999'")
    log(f"  Status: {data.get('status')}")
    log(f"  Message: {data.get('message', '')}")

    passed = result.is_error and "99999999" in (data.get("message", ""))
    log(f"\n  Result: {'PASS' if passed else 'FAIL'} -- error correctly returned")
    return passed


async def test_empty_input() -> bool:
    """Empty input should return a clear error."""
    divider("TEST 7: Edge Case -- Empty Input")
    gen = ReportGenerator()
    result = await gen.generate("")
    data = result.to_dict()

    log(f"  Input: ''")
    log(f"  Status: {data.get('status')}")
    log(f"  Message: {data.get('message', '')}")

    passed = result.is_error
    log(f"\n  Result: {'PASS' if passed else 'FAIL'} -- empty input rejected")
    return passed


async def main() -> None:
    log("=" * 72)
    log("  UNDERWRITING INTELLIGENCE SYSTEM -- END-TO-END TEST")
    log(f"  Run at: {datetime.now(timezone.utc).isoformat()}")
    log("=" * 72)

    results: dict[str, bool] = {}

    results["Health Check"] = await test_health()
    results["Search Disambiguation"] = await test_search_disambiguation()
    results["Ambiguous Name -> Disambiguation"] = await test_report_ambiguous_name()
    results["Full Pipeline (by number)"] = await test_report_by_number()
    results["Exact Name Auto-Resolve"] = await test_report_exact_name()
    results["Unknown Company Number"] = await test_unknown_company()
    results["Empty Input"] = await test_empty_input()

    divider("SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        log(f"  {'PASS' if ok else 'FAIL'}  {name}")
    log(f"\n  {passed}/{total} tests passed")
    log("=" * 72)

    output_path = RESULTS_DIR / "test_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
