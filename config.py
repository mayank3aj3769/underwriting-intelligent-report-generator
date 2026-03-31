import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Primary: Companies House ---
    COMPANIES_HOUSE_API_KEY: str = os.getenv("COMPANIES_HOUSE_API_KEY", "")
    COMPANIES_HOUSE_API_URL: str = os.getenv(
        "COMPANIES_HOUSE_API_URL",
        "https://api-sandbox.company-information.service.gov.uk",
    )

    # --- Search: SerpAPI (required) ---
    SERP_API_KEY: str = os.getenv("SERP_API_KEY", "")

    # --- LLM synthesis: OpenAI (required) ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # --- Agent behaviour ---
    MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "8"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

    USER_AGENT: str = (
        "UnderwritingIntelligenceBot/1.0 "
        "(+https://github.com/underwriting-intel)"
    )

    @property
    def has_companies_house_key(self) -> bool:
        return bool(self.COMPANIES_HOUSE_API_KEY)

    @property
    def has_serp_api_key(self) -> bool:
        return bool(self.SERP_API_KEY)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.OPENAI_API_KEY)


settings = Settings()
