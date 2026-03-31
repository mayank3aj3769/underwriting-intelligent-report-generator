"""Top-level report generation service.

Handles intent classification, entity resolution (with disambiguation),
then invokes the LangGraph pipeline to gather data and synthesise the report.
Also supports follow-up questions on generated reports.
"""

import logging

from agents.graph import pipeline
from schemas.company import CompanyProfile, CompanySearchResponse
from schemas.report import UnderwritingReport
from services.entity_resolver import EntityResolver
from services.followup_handler import FollowUpHandler, FollowUpResult
from services.intent_classifier import IntentClassifier

logger = logging.getLogger(__name__)


class ReportGenerationResult:
    """Wrapper for the report generation outcome.

    Exactly one of `report`, `disambiguation`, `rejection`, `follow_up`, or `error` will be set.
    """

    def __init__(
        self,
        report: UnderwritingReport | None = None,
        disambiguation: CompanySearchResponse | None = None,
        error: str | None = None,
        rejection: str | None = None,
        follow_up: FollowUpResult | None = None,
    ) -> None:
        self.report = report
        self.disambiguation = disambiguation
        self.error = error
        self.rejection = rejection
        self.follow_up = follow_up

    @property
    def needs_disambiguation(self) -> bool:
        return (
            self.disambiguation is not None
            and self.disambiguation.disambiguation_required
        )

    @property
    def is_rejected(self) -> bool:
        return self.rejection is not None

    @property
    def is_follow_up(self) -> bool:
        return self.follow_up is not None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict:
        if self.report:
            return {
                "status": "success",
                "report": self.report.model_dump(),
            }
        if self.follow_up:
            return {
                "status": "follow_up",
                "answer": self.follow_up.answer,
                "used_web_search": self.follow_up.used_web_search,
            }
        if self.rejection:
            return {
                "status": "rejected",
                "message": self.rejection,
            }
        if self.disambiguation:
            return {
                "status": "disambiguation_required",
                "message": (
                    f"Multiple companies match '{self.disambiguation.query}'. "
                    "Please select one by providing the company number."
                ),
                "total_results": self.disambiguation.total_results,
                "candidates": [
                    c.model_dump() for c in self.disambiguation.candidates
                ],
            }
        return {
            "status": "error",
            "message": self.error or "Unknown error",
        }


class ReportGenerator:
    """Orchestrates intent classification → entity resolution → LangGraph pipeline → report."""

    def __init__(self) -> None:
        self.resolver = EntityResolver()
        self.classifier = IntentClassifier()
        self.followup_handler = FollowUpHandler()

    async def generate(
        self,
        identifier: str,
        report_context: dict | None = None,
    ) -> ReportGenerationResult:
        identifier = identifier.strip()
        if not identifier:
            return ReportGenerationResult(error="Empty identifier provided")

        has_active_report = bool(report_context)
        logger.info("Classifying intent for: %s (active_report=%s)", identifier, has_active_report)

        intent = await self.classifier.classify(identifier, has_active_report=has_active_report)

        if intent.is_follow_up and report_context:
            logger.info("Handling follow-up question: '%s'", identifier)
            result = await self.followup_handler.answer(identifier, report_context)
            return ReportGenerationResult(follow_up=result)

        if intent.is_rejected:
            logger.info("Query rejected (out of scope): '%s'", identifier)
            return ReportGenerationResult(rejection=intent.rejection_message)

        if not intent.is_report_request:
            return ReportGenerationResult(rejection=intent.rejection_message)

        resolved_identifier = intent.company_identifier or identifier
        logger.info("Starting report generation for: %s", resolved_identifier)

        profile, search_result = await self.resolver.resolve(resolved_identifier)

        if search_result and search_result.disambiguation_required:
            logger.info(
                "Disambiguation needed: %d candidates for '%s'",
                search_result.total_results, resolved_identifier,
            )
            return ReportGenerationResult(disambiguation=search_result)

        if not profile:
            msg = f"Could not find company: '{resolved_identifier}'"
            if search_result and search_result.total_results == 0:
                msg = f"No companies found matching '{resolved_identifier}'"
            logger.warning(msg)
            return ReportGenerationResult(error=msg)

        return await self._run_pipeline(profile)

    async def generate_by_number(self, company_number: str) -> ReportGenerationResult:
        company_number = company_number.strip().upper()
        if not company_number:
            return ReportGenerationResult(error="Empty company number provided")

        profile = await self.resolver.resolve_by_number(company_number)
        if not profile:
            return ReportGenerationResult(
                error=f"No company found with number '{company_number}'"
            )

        return await self._run_pipeline(profile)

    async def _run_pipeline(self, profile: CompanyProfile) -> ReportGenerationResult:
        logger.info(
            "Invoking LangGraph pipeline for %s (%s)",
            profile.company_name, profile.company_number,
        )

        initial_state = {
            "company_number": profile.company_number,
            "company_name": profile.company_name,
            "company_profile_text": "",
            "company_metadata": {},
            "search_queries": [],
            "search_results": [],
            "search_summary": "",
            "evidence_metrics": {},
            "iteration_count": 0,
            "sufficiency_flag": False,
            "final_report": {},
            "errors": [],
        }

        try:
            result = await pipeline.ainvoke(initial_state)
        except Exception as e:
            logger.exception("Pipeline failed for %s", profile.company_number)
            return ReportGenerationResult(
                error=f"Pipeline execution failed: {e}"
            )

        errors = result.get("errors", [])
        if errors:
            logger.warning("Pipeline completed with errors: %s", errors)

        report_data = result.get("final_report", {})
        if not report_data:
            return ReportGenerationResult(
                error="Pipeline produced no report. Errors: " + "; ".join(errors)
            )

        try:
            report = UnderwritingReport(**report_data)
        except Exception as e:
            logger.error("Failed to deserialise report: %s", e)
            return ReportGenerationResult(
                error=f"Report deserialisation failed: {e}"
            )

        logger.info(
            "Report generated with %d evidence items for %s",
            report.raw_evidence_count, profile.company_name,
        )
        return ReportGenerationResult(report=report)
