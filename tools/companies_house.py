import logging
from typing import Optional

import httpx

from config import settings
from schemas.company import CompanyCandidate, CompanyProfile, Officer, CompanySearchResponse
from schemas.evidence import Evidence, SourceType
from tools.base import BaseTool

logger = logging.getLogger(__name__)


class CompaniesHouseTool(BaseTool):
    name = "companies_house"
    description = "Fetch company data from the UK Companies House API"

    def __init__(self) -> None:
        self._api_key = settings.COMPANIES_HOUSE_API_KEY
        self._base_url = settings.COMPANIES_HOUSE_API_URL
        self._timeout = settings.REQUEST_TIMEOUT

    def _auth(self) -> httpx.BasicAuth:
        return httpx.BasicAuth(username=self._api_key, password="")

    async def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        if not self._api_key:
            logger.warning("Companies House API key not configured")
            return None
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self._base_url}{path}",
                    params=params,
                    auth=self._auth(),
                )
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("Companies House API error %s: %s", e.response.status_code, e)
            return None
        except httpx.RequestError as e:
            logger.error("Companies House request failed: %s", e)
            return None

    async def search_companies(self, query: str) -> CompanySearchResponse:
        data = await self._get("/search/companies", params={"q": query, "items_per_page": 10})
        if not data:
            return CompanySearchResponse(
                query=query, total_results=0, candidates=[], disambiguation_required=False,
            )

        candidates: list[CompanyCandidate] = []
        for item in data.get("items", []):
            addr = item.get("address", {})
            candidates.append(
                CompanyCandidate(
                    company_number=item.get("company_number", ""),
                    company_name=item.get("title", ""),
                    company_status=item.get("company_status"),
                    company_type=item.get("company_type"),
                    date_of_creation=item.get("date_of_creation"),
                    registered_office_address=addr if addr else None,
                    snippet=item.get("snippet"),
                )
            )

        total = data.get("total_results", len(candidates))
        needs_disambiguation = len(candidates) > 1
        return CompanySearchResponse(
            query=query,
            total_results=total,
            candidates=candidates,
            disambiguation_required=needs_disambiguation,
        )

    async def get_company_profile(self, company_number: str) -> Optional[CompanyProfile]:
        data = await self._get(f"/company/{company_number}")
        if not data:
            return None
        return CompanyProfile(
            company_number=data.get("company_number", company_number),
            company_name=data.get("company_name", ""),
            company_status=data.get("company_status"),
            company_type=data.get("type"),
            date_of_creation=data.get("date_of_creation"),
            date_of_cessation=data.get("date_of_cessation"),
            sic_codes=data.get("sic_codes", []),
            registered_office_address=data.get("registered_office_address"),
            accounts=data.get("accounts"),
            confirmation_statement=data.get("confirmation_statement"),
            has_charges=data.get("has_charges"),
            has_insolvency_history=data.get("has_insolvency_history"),
            jurisdiction=data.get("jurisdiction"),
            raw_data=data,
        )

    async def get_officers(self, company_number: str) -> list[Officer]:
        data = await self._get(f"/company/{company_number}/officers")
        if not data:
            return []
        officers: list[Officer] = []
        for item in data.get("items", []):
            officers.append(
                Officer(
                    name=item.get("name", "Unknown"),
                    officer_role=item.get("officer_role", "unknown"),
                    appointed_on=item.get("appointed_on"),
                    resigned_on=item.get("resigned_on"),
                    nationality=item.get("nationality"),
                    occupation=item.get("occupation"),
                    country_of_residence=item.get("country_of_residence"),
                )
            )
        return officers

    async def execute(self, **kwargs) -> list[Evidence]:
        """Gather evidence from Companies House for a given company_number."""
        company_number = kwargs.get("company_number", "")
        if not company_number:
            return []

        evidences: list[Evidence] = []
        profile = await self.get_company_profile(company_number)
        if profile:
            sic_text = ", ".join(profile.sic_codes) if profile.sic_codes else "not listed"
            address_parts = []
            if profile.registered_office_address:
                for key in ("address_line_1", "address_line_2", "locality", "postal_code", "country"):
                    val = profile.registered_office_address.get(key)
                    if val:
                        address_parts.append(val)
            address_str = ", ".join(address_parts) if address_parts else "not available"

            content = (
                f"Company: {profile.company_name}\n"
                f"Number: {profile.company_number}\n"
                f"Status: {profile.company_status or 'unknown'}\n"
                f"Type: {profile.company_type or 'unknown'}\n"
                f"Incorporated: {profile.date_of_creation or 'unknown'}\n"
                f"SIC Codes: {sic_text}\n"
                f"Registered Address: {address_str}\n"
                f"Has Charges: {profile.has_charges}\n"
                f"Insolvency History: {profile.has_insolvency_history}\n"
            )
            evidences.append(
                Evidence(
                    source_type=SourceType.COMPANIES_HOUSE,
                    source_url=f"https://find-and-update.company-information.service.gov.uk/company/{company_number}",
                    title=f"Companies House Profile: {profile.company_name}",
                    content=content,
                    confidence=1.0,
                    metadata={
                        "company_number": profile.company_number,
                        "company_name": profile.company_name,
                        "sic_codes": profile.sic_codes,
                        "company_status": profile.company_status,
                        "date_of_creation": profile.date_of_creation,
                        "has_charges": profile.has_charges,
                        "has_insolvency_history": profile.has_insolvency_history,
                    },
                )
            )

        officers = await self.get_officers(company_number)
        if officers:
            active = [o for o in officers if not o.resigned_on]
            lines = [f"- {o.name} ({o.officer_role})" for o in active[:10]]
            content = f"Active officers ({len(active)} total):\n" + "\n".join(lines)
            evidences.append(
                Evidence(
                    source_type=SourceType.COMPANIES_HOUSE,
                    source_url=f"https://find-and-update.company-information.service.gov.uk/company/{company_number}/officers",
                    title=f"Officers of {profile.company_name if profile else company_number}",
                    content=content,
                    confidence=1.0,
                    metadata={"officer_count": len(active)},
                )
            )

        return evidences
