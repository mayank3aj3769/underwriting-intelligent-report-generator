"""SerpAPI client — the sole search backend. No fallbacks.

Returns structured snippets, knowledge-graph data, and news results.
No DOM parsing, no scraping.
"""

import asyncio
import logging

from config import settings

logger = logging.getLogger(__name__)


class SerpAPIClient:
    """Thin async wrapper around the SerpAPI Google Search API."""

    async def search(self, query: str, num: int = 6) -> list[dict]:
        """Run a Google web search and return normalised results."""
        from serpapi import GoogleSearch

        params = {
            "q": query,
            "api_key": settings.SERP_API_KEY,
            "num": num,
            "gl": "uk",
            "hl": "en",
        }
        data = await asyncio.to_thread(self._run_search, params)
        results: list[dict] = []

        for item in data.get("organic_results", [])[:num]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
                "source": item.get("source", ""),
                "position": item.get("position"),
                "type": "web",
            })

        kg = data.get("knowledge_graph")
        if kg:
            results.append({
                "title": kg.get("title", "Knowledge Graph"),
                "snippet": kg.get("description", ""),
                "url": kg.get("website", kg.get("source", {}).get("link", "")),
                "source": "Google Knowledge Graph",
                "type": "knowledge_graph",
                "kg_type": kg.get("type"),
            })

        return results

    async def search_news(self, query: str, num: int = 8) -> list[dict]:
        """Run a Google News search and return normalised results."""
        from serpapi import GoogleSearch

        params = {
            "q": query,
            "api_key": settings.SERP_API_KEY,
            "tbm": "nws",
            "num": num,
            "gl": "uk",
            "hl": "en",
        }
        data = await asyncio.to_thread(self._run_search, params)
        results: list[dict] = []

        for item in data.get("news_results", [])[:num]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
                "source": item.get("source", ""),
                "date": item.get("date", ""),
                "type": "news",
            })

        return results

    @staticmethod
    def _run_search(params: dict) -> dict:
        from serpapi import GoogleSearch

        search = GoogleSearch(params)
        return search.get_dict()
