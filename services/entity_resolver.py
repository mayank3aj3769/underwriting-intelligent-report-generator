import logging
import re

from schemas.company import CompanyProfile, CompanySearchResponse
from tools.companies_house import CompaniesHouseTool

logger = logging.getLogger(__name__)

CH_NUMBER_PATTERN = re.compile(r"^[A-Z]{0,2}\d{6,8}$", re.IGNORECASE)


class EntityResolver:
    """Resolves a company identifier (name or number) to a Companies House profile.

    If the input looks like a registration number, fetches directly.
    If it is a name and matches multiple companies, returns candidates
    for disambiguation instead of proceeding blindly.
    """

    def __init__(self) -> None:
        self.ch_tool = CompaniesHouseTool()

    def is_company_number(self, identifier: str) -> bool:
        return bool(CH_NUMBER_PATTERN.match(identifier.strip()))

    async def resolve_by_number(self, company_number: str) -> CompanyProfile | None:
        company_number = company_number.strip().upper()
        profile = await self.ch_tool.get_company_profile(company_number)
        if not profile:
            logger.warning("No company found for number %s", company_number)
        return profile

    async def search_by_name(self, company_name: str) -> CompanySearchResponse:
        result = await self.ch_tool.search_companies(company_name.strip())
        if result.total_results == 0:
            logger.info("No results for '%s'", company_name)
        elif result.total_results == 1:
            result.disambiguation_required = False
        else:
            active = [c for c in result.candidates if c.company_status == "active"]
            if len(active) == 1:
                result.candidates = active
                result.disambiguation_required = False
                result.total_results = 1
            else:
                exact = [
                    c for c in result.candidates
                    if c.company_name.upper() == company_name.strip().upper()
                    and c.company_status == "active"
                ]
                if len(exact) == 1:
                    result.candidates = exact
                    result.disambiguation_required = False
                    result.total_results = 1
                else:
                    result.disambiguation_required = True
        return result

    async def resolve(self, identifier: str) -> tuple[CompanyProfile | None, CompanySearchResponse | None]:
        """Resolve an identifier.

        Returns:
            (profile, None) if unambiguous resolution succeeds.
            (None, search_response) if disambiguation is needed.
            (None, None) if nothing is found.
        """
        identifier = identifier.strip()

        if self.is_company_number(identifier):
            profile = await self.resolve_by_number(identifier)
            return (profile, None)

        search_result = await self.search_by_name(identifier)

        if search_result.total_results == 0:
            return (None, search_result)

        if not search_result.disambiguation_required and search_result.candidates:
            best = search_result.candidates[0]
            profile = await self.resolve_by_number(best.company_number)
            return (profile, None)

        return (None, search_result)
