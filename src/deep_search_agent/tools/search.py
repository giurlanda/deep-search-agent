"""SearxNG-backed web search tool.

Exposes :func:`create_searxng_search_tool`, a factory that builds a
LangChain tool querying a SearxNG instance through its JSON API
(``GET <base_url>/search?q=...&format=json``).

Errors (connection refused, timeouts, non-2xx responses) are returned as an
``ERROR: ...`` string instead of raising, so the calling agent can reformulate
the query or reroute without crashing the whole research flow.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx
from langchain_core.tools import BaseTool, tool

DEFAULT_SEARXNG_BASE_URL = "http://localhost:8888"


def _format_result(index: int, item: dict) -> str:
    """Render a single SearxNG result entry as a markdown block.

    Args:
        index: 1-based position of the result in the response.
        item: Raw result dict from the SearxNG JSON API.

    Returns:
        A markdown fragment with title, URL, snippet, and available metadata.
    """
    title = item.get("title") or "(no title)"
    url = item.get("url") or ""
    snippet = (item.get("content") or "").strip()
    lines = [f"{index}. **{title}**", f"   URL: {url}"]
    if snippet:
        lines.append(f"   Snippet: {snippet}")
    if item.get("publishedDate"):
        lines.append(f"   Published: {item['publishedDate']}")
    if item.get("engine"):
        lines.append(f"   Engine: {item['engine']}")
    return "\n".join(lines)


def create_searxng_search_tool(
    *,
    base_url: str = DEFAULT_SEARXNG_BASE_URL,
    engines: Sequence[str] | None = None,
    timeout: float = 15.0,
    max_results: int = 5,
) -> BaseTool:
    """Build a web-search tool backed by a SearxNG instance.

    Args:
        base_url: Root URL of the SearxNG instance
            (e.g. ``"http://localhost:8888"``). Trailing slashes are ignored.
        engines: Optional list of SearxNG engine names to restrict the search
            to (e.g. ``["duckduckgo", "wikipedia"]``). ``None`` lets SearxNG
            use its default engine set.
        timeout: Per-request timeout in seconds.
        max_results: Maximum number of results returned per query.

    Returns:
        A LangChain tool named ``internet_search`` that takes a ``query``
        string and returns markdown-formatted results, or an ``ERROR: ...``
        string on failure.
    """
    search_url = f"{base_url.rstrip('/')}/search"
    engines_param = ",".join(engines) if engines else None

    @tool
    def internet_search(query: str) -> str:
        """Search the web via SearxNG and return the top results.

        Args:
            query: The search query. Be specific; reformulate and call again
                if the results are not relevant.

        Returns:
            Markdown list of results (title, URL, snippet, date, engine),
            or an ``ERROR: ...`` message if the search failed.
        """
        params: dict[str, str] = {"q": query, "format": "json"}
        if engines_param:
            params["engines"] = engines_param
        try:
            response = httpx.get(search_url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            return (
                f"ERROR: SearxNG returned HTTP {exc.response.status_code} "
                f"for query {query!r}. Try again later or reformulate."
            )
        except httpx.HTTPError as exc:
            return (
                f"ERROR: could not reach SearxNG at {search_url}: {exc}. "
                "Check connectivity or retry with a different query."
            )
        except ValueError:
            return (
                "ERROR: SearxNG response was not valid JSON. The instance may "
                "not have the JSON format enabled."
            )

        results = payload.get("results") or []
        if not results:
            return f"No results found for query {query!r}. Try a reformulation."
        formatted = [
            _format_result(i, item)
            for i, item in enumerate(results[:max_results], start=1)
        ]
        return "\n\n".join(formatted)

    return internet_search
