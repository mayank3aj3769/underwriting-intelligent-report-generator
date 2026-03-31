"""Follow-up question handler for conversational report interaction.

After a report is generated, users can ask follow-up questions.
The handler decides whether the answer exists in the report context
or requires a new web search, then synthesises a grounded answer.
"""

import json
import logging

from openai import AsyncOpenAI

from config import settings
from tools.search_api import SerpAPIClient

logger = logging.getLogger(__name__)


class FollowUpResult:
    """Result of a follow-up question."""

    def __init__(self, answer: str, used_web_search: bool = False, search_query: str = "") -> None:
        self.answer = answer
        self.used_web_search = used_web_search
        self.search_query = search_query


class FollowUpHandler:
    """Answers follow-up questions using the report context and optional web search."""

    def __init__(self) -> None:
        self.search_client = SerpAPIClient()

    async def answer(self, question: str, report_context: dict) -> FollowUpResult:
        """Answer a follow-up question about the current report.

        Steps:
          1. Ask the LLM whether the report already contains the answer
          2. If yes — synthesise from report context
          3. If no — generate a search query, run SerpAPI, synthesise with both
        """
        company_name = report_context.get("company_name", "the company")
        readable_report = report_context.get("readable_report", "")

        needs_search, search_query = await self._assess_need(question, company_name, readable_report)

        if needs_search and search_query:
            logger.info("Follow-up requires web search: '%s'", search_query)
            search_results = await self._run_search(search_query)
            answer = await self._answer_with_search(
                question, company_name, readable_report, search_results,
            )
            return FollowUpResult(answer=answer, used_web_search=True, search_query=search_query)

        logger.info("Follow-up answered from report context")
        answer = await self._answer_from_report(question, company_name, readable_report)
        return FollowUpResult(answer=answer, used_web_search=False)

    async def _assess_need(
        self, question: str, company_name: str, readable_report: str,
    ) -> tuple[bool, str]:
        """Determine whether a web search is needed and generate a query if so."""
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        prompt = (
            f"You have an underwriting intelligence report for **{company_name}**.\n\n"
            f"REPORT CONTENT (summary):\n{readable_report[:4000]}\n\n"
            f"The user asks: \"{question}\"\n\n"
            "Can this question be answered FULLY from the report above?\n"
            "Return JSON:\n"
            "{\n"
            "  \"can_answer_from_report\": true/false,\n"
            "  \"reasoning\": \"brief explanation\",\n"
            "  \"search_query\": \"Google search query if web search needed, else empty string\"\n"
            "}"
        )

        try:
            resp = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You decide whether a follow-up question about a company "
                            "can be answered from an existing report or needs a web search. "
                            "Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            can_answer = data.get("can_answer_from_report", True)
            search_query = data.get("search_query", "").strip()
            logger.info(
                "Follow-up assessment: can_answer=%s, search_query='%s', reasoning='%s'",
                can_answer, search_query, data.get("reasoning", ""),
            )
            return (not can_answer, search_query)
        except Exception as e:
            logger.warning("Follow-up assessment failed: %s — answering from report", e)
            return (False, "")

    async def _run_search(self, query: str) -> list[dict]:
        """Execute a single web search for the follow-up question."""
        try:
            results = await self.search_client.search(query, num=5)
            logger.info("Follow-up search returned %d results", len(results))
            return results
        except Exception as e:
            logger.warning("Follow-up search failed: %s", e)
            return []

    async def _answer_from_report(
        self, question: str, company_name: str, readable_report: str,
    ) -> str:
        """Generate an answer using only the existing report content."""
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        prompt = (
            f"You are an underwriting intelligence assistant. The user has a report "
            f"for **{company_name}** and is asking a follow-up question.\n\n"
            f"REPORT:\n{readable_report}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Answer the question using ONLY information from the report above.\n"
            "- Be specific and reference the report sections where relevant\n"
            "- If the report doesn't contain enough information, say so clearly\n"
            "- Keep the answer concise but thorough\n"
            "- Use markdown formatting for readability"
        )

        try:
            resp = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful underwriting intelligence assistant. "
                            "Answer follow-up questions about company reports accurately "
                            "and concisely. Only use information from the provided report."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or "I couldn't generate an answer."
        except Exception as e:
            logger.error("Follow-up answer generation failed: %s", e)
            return f"Sorry, I encountered an error generating the answer: {e}"

    async def _answer_with_search(
        self,
        question: str,
        company_name: str,
        readable_report: str,
        search_results: list[dict],
    ) -> str:
        """Generate an answer combining report context and new search results."""
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        search_block = self._format_search_results(search_results)

        prompt = (
            f"You are an underwriting intelligence assistant. The user has a report "
            f"for **{company_name}** and is asking a follow-up question that requires "
            f"additional information beyond the report.\n\n"
            f"EXISTING REPORT:\n{readable_report[:3000]}\n\n"
            f"ADDITIONAL SEARCH RESULTS:\n{search_block}\n\n"
            f"USER QUESTION: {question}\n\n"
            "Answer the question by combining the report context with the new search results.\n"
            "- Cite sources with [Source: URL] where applicable\n"
            "- Clearly distinguish between report-based and newly-found information\n"
            "- If information conflicts, present both sides\n"
            "- Keep the answer concise but thorough\n"
            "- Use markdown formatting for readability"
        )

        try:
            resp = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful underwriting intelligence assistant. "
                            "Answer follow-up questions using both the existing report "
                            "and new search results. Cite sources for new information."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or "I couldn't generate an answer."
        except Exception as e:
            logger.error("Follow-up answer with search failed: %s", e)
            return f"Sorry, I encountered an error generating the answer: {e}"

    @staticmethod
    def _format_search_results(results: list[dict]) -> str:
        if not results:
            return "(no additional search results)"
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.get('title', '')}\n"
                f"    URL: {r.get('url', 'N/A')}\n"
                f"    Snippet: {r.get('snippet', '')}"
            )
        return "\n".join(lines)
