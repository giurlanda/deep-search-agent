"""SearxNG-backed web search tool.

Exposes :func:`create_searxng_search_tool`, a factory that builds a
LangChain tool querying a SearxNG instance through its JSON API
(``GET <base_url>/search?q=...&format=json``).

Errors (connection refused, timeouts, non-2xx responses) are returned as an
``ERROR: ...`` string instead of raising, so the calling agent can reformulate
the query or reroute without crashing the whole research flow.

The tool can be shared by several sub-agents that run concurrently through
deepagents' thread pool, so two throttling primitives guard the SearxNG
instance and are safe to call from multiple threads:

- :class:`_MinIntervalRateLimiter` spaces successive requests by at least a
  fixed interval (``searxng_rate_limit``).
- :class:`SearchBudget` caps the number of search operations allowed in a
  single research cycle (``searxng_budget``); it is reset at each cycle
  boundary from outside the tool.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import httpx
from langchain_core.tools import BaseTool, tool

DEFAULT_SEARXNG_BASE_URL = "http://localhost:8888"


class SearchBudget:
    """Thread-safe cap on the number of search operations per research cycle.

    The same :func:`create_searxng_search_tool` instance may be invoked
    concurrently by several sub-agents (deepagents runs tool calls through a
    thread pool), so consumption is serialized with a lock. The budget is
    meant to be reset at each research-cycle boundary (see
    :class:`~deep_search_agent.middleware.SearchBudgetResetMiddleware`).

    Args:
        limit: Maximum number of successful consumptions before the budget is
            exhausted. ``None`` means unlimited (every consumption succeeds).
    """

    def __init__(self, limit: int | None) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._used = 0

    def try_consume(self) -> bool:
        """Consume one unit of budget.

        Returns:
            ``True`` if a unit was available (and has been consumed),
            ``False`` if the budget for the current cycle is exhausted.
        """
        if self._limit is None:
            return True
        with self._lock:
            if self._used >= self._limit:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one previously consumed unit.

        Used when a consumption did not translate into an actual request
        (e.g. the rate limiter aborted before the HTTP call was made), so the
        budget keeps counting only searches that were really dispatched.
        """
        if self._limit is None:
            return
        with self._lock:
            if self._used > 0:
                self._used -= 1

    def reset(self) -> None:
        """Restore the full budget for a new research cycle."""
        with self._lock:
            self._used = 0


class _MinIntervalRateLimiter:
    """Thread-safe limiter enforcing a minimum interval between requests.

    Callers reserve the next free slot via :meth:`acquire`; concurrent
    threads are spaced out by at least ``min_interval`` seconds. Reservation
    happens under a lock but the returned wait is slept *outside* the lock, so
    threads do not serialize on the sleep itself.

    Args:
        min_interval: Minimum number of seconds between two successive
            requests. Must be positive.
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_available = 0.0

    def acquire(self, max_wait: float) -> float | None:
        """Reserve the next slot and report how long to wait for it.

        Args:
            max_wait: Upper bound (seconds) on an acceptable wait. If the next
                slot is farther away than this, no slot is reserved.

        Returns:
            The number of seconds the caller must ``sleep`` before proceeding
            (``0.0`` if the slot is available now), or ``None`` if the wait
            would exceed ``max_wait`` (the caller should abort without a
            request).
        """
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_available)
            wait = scheduled - now
            if wait > max_wait:
                return None
            self._next_available = scheduled + self._min_interval
            return wait


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
    min_request_interval: float | None = None,
    budget: SearchBudget | None = None,
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
        min_request_interval: Minimum number of seconds between two SearxNG
            requests, enforced by a thread-safe limiter shared across the
            (possibly concurrent) callers of the returned tool. ``None``
            disables rate limiting. When a request would have to wait longer
            than ``timeout`` for a free slot, the tool returns an
            ``ERROR: ...`` string instead of performing it.
        budget: Optional :class:`SearchBudget` capping the number of search
            operations; when exhausted the tool returns an ``ERROR: ...``
            string telling the model no budget is left. The caller owns the
            budget and is responsible for resetting it (e.g. per research
            cycle). ``None`` means unlimited.

    Returns:
        A LangChain tool named ``internet_search`` that takes a ``query``
        string and returns markdown-formatted results, or an ``ERROR: ...``
        string on failure.
    """
    search_url = f"{base_url.rstrip('/')}/search"
    engines_param = ",".join(engines) if engines else None
    rate_limiter = (
        _MinIntervalRateLimiter(min_request_interval) if min_request_interval else None
    )

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
        # Budget gate first: cheap, and lets the model know immediately when
        # it has run out of searches without waiting on the rate limiter.
        if budget is not None and not budget.try_consume():
            return (
                "ERROR: search budget exhausted for this research cycle. No "
                "further searches can be run now; rely on the findings already "
                "gathered or wait for the next cycle."
            )
        # Rate limit second: reserve a slot spaced from other concurrent
        # searches. If the slot is farther away than the request timeout,
        # abort and refund the budget so it counts only dispatched searches.
        if rate_limiter is not None:
            wait = rate_limiter.acquire(timeout)
            if wait is None:
                if budget is not None:
                    budget.refund()
                return (
                    "ERROR: SearxNG rate limit would require waiting longer "
                    f"than the {timeout}s timeout for query {query!r}. Try "
                    "again shortly or reduce the number of parallel searches."
                )
            if wait > 0:
                time.sleep(wait)

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
