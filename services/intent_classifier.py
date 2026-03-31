"""Intent classifier — gatekeeper that routes user input to the
correct handler.

Supports three intents:
  - report:       Generate a new report (company name or number)
  - follow_up:    Ask a follow-up question about the current report
  - out_of_scope: Anything else (rejected)

Uses a two-tier approach:
  1. Fast heuristic check (regex for company numbers, obvious non-queries)
  2. LLM classification for ambiguous inputs
"""

import json
import logging
import re

from openai import AsyncOpenAI

from config import settings

logger = logging.getLogger(__name__)

CH_NUMBER_PATTERN = re.compile(r"^[A-Z]{0,2}\d{6,8}$", re.IGNORECASE)

_OBVIOUS_REJECT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^(hi|hello|hey|yo|sup)\s*[.!?]?\s*$",
        r"^(translate|convert|calculate|compute|solve)\b",
        r"\b(weather|recipe|joke|poem|song|story|essay|script|email)\b",
        r"^(thank|thanks|cheers)\s*[.!?]?\s*$",
        r"^(yes|no|ok|okay|sure|nope|yep)\s*$",
    ]
]

_REJECTION_MESSAGE = (
    "I'm the **Underwriting Intelligence Report Generator** — I'm designed "
    "exclusively to produce intelligence reports on UK companies.\n\n"
    "I'm not authorised to perform other tasks such as general knowledge "
    "questions, writing, coding, or conversation.\n\n"
    "**What I can do:**\n"
    "- Generate a report from a **company name** (e.g. *REVOLUT LTD*)\n"
    "- Generate a report from a **Companies House number** (e.g. *08804411*)\n"
    "- Answer **follow-up questions** about a generated report\n\n"
    "Please enter a UK company name or registration number to get started."
)


class IntentResult:
    """Result of intent classification."""

    REPORT = "report"
    FOLLOW_UP = "follow_up"
    OUT_OF_SCOPE = "out_of_scope"

    def __init__(
        self,
        intent: str,
        company_identifier: str = "",
        rejection_message: str = "",
    ) -> None:
        self.intent = intent
        self.company_identifier = company_identifier
        self.rejection_message = rejection_message

    @property
    def is_report_request(self) -> bool:
        return self.intent == self.REPORT

    @property
    def is_follow_up(self) -> bool:
        return self.intent == self.FOLLOW_UP

    @property
    def is_rejected(self) -> bool:
        return self.intent == self.OUT_OF_SCOPE

    @staticmethod
    def accept(identifier: str) -> "IntentResult":
        return IntentResult(intent=IntentResult.REPORT, company_identifier=identifier)

    @staticmethod
    def follow_up() -> "IntentResult":
        return IntentResult(intent=IntentResult.FOLLOW_UP)

    @staticmethod
    def reject(message: str = "") -> "IntentResult":
        return IntentResult(
            intent=IntentResult.OUT_OF_SCOPE,
            rejection_message=message or _REJECTION_MESSAGE,
        )


class IntentClassifier:
    """Classifies user input as a report request, follow-up, or out-of-scope query."""

    async def classify(self, user_input: str, has_active_report: bool = False) -> IntentResult:
        text = user_input.strip()

        if not text:
            return IntentResult.reject()

        if CH_NUMBER_PATTERN.match(text):
            logger.info("Intent: company number detected (fast path): %s", text)
            return IntentResult.accept(text)

        if len(text) <= 2:
            return IntentResult.reject()

        if not has_active_report:
            for pattern in _OBVIOUS_REJECT_PATTERNS:
                if pattern.search(text):
                    logger.info("Intent: rejected by heuristic pattern: '%s'", text)
                    return IntentResult.reject()

        return await self._llm_classify(text, has_active_report)

    async def _llm_classify(self, text: str, has_active_report: bool) -> IntentResult:
        """Use the LLM for ambiguous inputs that pass heuristic checks."""
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        context_block = ""
        if has_active_report:
            context_block = (
                "\nIMPORTANT CONTEXT: The user currently has an active company report "
                "open in their session. They may be asking a follow-up question about it.\n"
                "Follow-up questions include things like:\n"
                "- Asking for more detail about a section (competitors, revenue, risks)\n"
                "- Asking about something related to the company\n"
                "- Requesting clarification or deeper analysis\n"
                "- Asking 'what about X?' or 'tell me more about Y'\n\n"
            )

        intents_block = (
            "Possible intents:\n"
            "- \"report\": User wants to generate a NEW report for a company\n"
        )
        if has_active_report:
            intents_block += (
                "- \"follow_up\": User is asking a question about the currently active report\n"
            )
        intents_block += "- \"out_of_scope\": Anything else\n"

        prompt = (
            "You are an intent classifier for an underwriting intelligence report system.\n\n"
            "The system generates intelligence reports on UK companies. It accepts:\n"
            "- A UK company name (e.g. \"Revolut\", \"Barclays PLC\", \"Tesco\")\n"
            "- A Companies House registration number (e.g. \"08804411\", \"SC123456\")\n"
            "- A request to generate a report (e.g. \"generate report for Revolut\")\n"
            f"{context_block}\n"
            f"{intents_block}\n"
            f"User input: \"{text}\"\n\n"
            "Classify this input. Return JSON:\n"
            "{\n"
            "  \"intent\": \"report\" or \"follow_up\" or \"out_of_scope\",\n"
            "  \"company_identifier\": \"extracted company name/number if intent is report, else empty string\"\n"
            "}"
        )

        try:
            resp = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict intent classifier. Classify user input into "
                            "the correct intent category. Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=150,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            intent = data.get("intent", "out_of_scope")
            identifier = data.get("company_identifier", "").strip()

            if intent == "report" and identifier:
                logger.info("Intent: LLM classified as report request for '%s'", identifier)
                return IntentResult.accept(identifier)
            elif intent == "report":
                logger.info("Intent: LLM classified as report but no identifier, using raw input")
                return IntentResult.accept(text)
            elif intent == "follow_up" and has_active_report:
                logger.info("Intent: LLM classified as follow_up: '%s'", text)
                return IntentResult.follow_up()
            else:
                logger.info("Intent: LLM classified as out_of_scope: '%s'", text)
                return IntentResult.reject()

        except Exception as e:
            logger.warning("Intent classification LLM call failed: %s — allowing through", e)
            if has_active_report:
                return IntentResult.follow_up()
            return IntentResult.accept(text)
