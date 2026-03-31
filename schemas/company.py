from pydantic import BaseModel, Field
from typing import Optional


class CompanyCandidate(BaseModel):
    """A company match returned from search."""

    company_number: str
    company_name: str
    company_status: Optional[str] = None
    company_type: Optional[str] = None
    date_of_creation: Optional[str] = None
    registered_office_address: Optional[dict] = None
    snippet: Optional[str] = None
    match_score: Optional[float] = None


class CompanyProfile(BaseModel):
    """Full profile from Companies House."""

    company_number: str
    company_name: str
    company_status: Optional[str] = None
    company_type: Optional[str] = None
    date_of_creation: Optional[str] = None
    date_of_cessation: Optional[str] = None
    sic_codes: list[str] = Field(default_factory=list)
    registered_office_address: Optional[dict] = None
    accounts: Optional[dict] = None
    confirmation_statement: Optional[dict] = None
    has_charges: Optional[bool] = None
    has_insolvency_history: Optional[bool] = None
    jurisdiction: Optional[str] = None
    type_description: Optional[str] = None
    raw_data: dict = Field(default_factory=dict)


class Officer(BaseModel):
    """A company officer (director, secretary, etc.)."""

    name: str
    officer_role: str
    appointed_on: Optional[str] = None
    resigned_on: Optional[str] = None
    nationality: Optional[str] = None
    occupation: Optional[str] = None
    country_of_residence: Optional[str] = None


class CompanySearchResponse(BaseModel):
    """Response from a company name search."""

    query: str
    total_results: int
    candidates: list[CompanyCandidate]
    disambiguation_required: bool

    @property
    def is_unique(self) -> bool:
        return len(self.candidates) == 1 and not self.disambiguation_required
