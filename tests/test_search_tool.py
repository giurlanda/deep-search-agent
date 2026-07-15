"""Unit tests for the SearxNG search tool (no network access)."""

from __future__ import annotations

import httpx

from deep_search_agent.tools import SearchBudget, create_searxng_search_tool
from deep_search_agent.tools.search import _MinIntervalRateLimiter
from tests.conftest import FakeResponse, make_searxng_payload


def test_returns_formatted_results(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(json_data=make_searxng_payload(3))
    tool = create_searxng_search_tool(base_url="http://searx.local:8888")

    output = tool.invoke({"query": "langchain deepagents"})

    assert "Result 1" in output
    assert "https://example.org/1" in output
    assert "Snippet 2" in output
    assert "duckduckgo" in output
    url, kwargs = fake_httpx_get.calls[0]
    assert url == "http://searx.local:8888/search"
    assert kwargs["params"]["q"] == "langchain deepagents"
    assert kwargs["params"]["format"] == "json"


def test_caps_results_at_max_results(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(json_data=make_searxng_payload(10))
    tool = create_searxng_search_tool(max_results=2)

    output = tool.invoke({"query": "q"})

    assert "Result 2" in output
    assert "Result 3" not in output


def test_engines_are_passed_as_csv(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(json_data=make_searxng_payload(1))
    tool = create_searxng_search_tool(engines=["duckduckgo", "wikipedia"])

    tool.invoke({"query": "q"})

    _, kwargs = fake_httpx_get.calls[0]
    assert kwargs["params"]["engines"] == "duckduckgo,wikipedia"


def test_no_engines_param_by_default(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(json_data=make_searxng_payload(1))
    tool = create_searxng_search_tool()

    tool.invoke({"query": "q"})

    _, kwargs = fake_httpx_get.calls[0]
    assert "engines" not in kwargs["params"]


def test_empty_results_returns_hint(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(json_data={"results": []})
    tool = create_searxng_search_tool()

    output = tool.invoke({"query": "xyzzy"})

    assert "No results" in output


def test_connection_error_returns_error_string(fake_httpx_get):
    fake_httpx_get.error = httpx.ConnectError("connection refused")
    tool = create_searxng_search_tool()

    output = tool.invoke({"query": "q"})

    assert output.startswith("ERROR:")
    assert "could not reach SearxNG" in output


def test_http_error_returns_error_string(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(status_code=429)
    tool = create_searxng_search_tool()

    output = tool.invoke({"query": "q"})

    assert output.startswith("ERROR:")
    assert "429" in output


def test_invalid_json_returns_error_string(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(json_data=None)
    tool = create_searxng_search_tool()

    output = tool.invoke({"query": "q"})

    assert output.startswith("ERROR:")
    assert "JSON" in output


# --- SearchBudget -------------------------------------------------------------


def test_budget_none_is_unlimited():
    budget = SearchBudget(None)

    assert all(budget.try_consume() for _ in range(100))


def test_budget_exhausts_after_limit():
    budget = SearchBudget(2)

    assert budget.try_consume() is True
    assert budget.try_consume() is True
    assert budget.try_consume() is False


def test_budget_refund_gives_back_a_unit():
    budget = SearchBudget(1)
    assert budget.try_consume() is True
    assert budget.try_consume() is False

    budget.refund()

    assert budget.try_consume() is True


def test_budget_reset_restores_full_budget():
    budget = SearchBudget(1)
    assert budget.try_consume() is True
    assert budget.try_consume() is False

    budget.reset()

    assert budget.try_consume() is True


# --- _MinIntervalRateLimiter --------------------------------------------------


def test_rate_limiter_first_slot_is_immediate():
    limiter = _MinIntervalRateLimiter(0.5)

    assert limiter.acquire(max_wait=10.0) == 0.0


def test_rate_limiter_spaces_successive_slots():
    limiter = _MinIntervalRateLimiter(0.5)

    limiter.acquire(max_wait=10.0)
    wait = limiter.acquire(max_wait=10.0)

    assert wait is not None
    assert 0.0 < wait <= 0.5


def test_rate_limiter_aborts_when_wait_exceeds_max():
    limiter = _MinIntervalRateLimiter(100.0)

    assert limiter.acquire(max_wait=1.0) == 0.0
    assert limiter.acquire(max_wait=1.0) is None


# --- Rate limit and budget through the tool -----------------------------------


def test_tool_budget_exhausted_returns_error_without_request(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(json_data=make_searxng_payload(1))
    tool = create_searxng_search_tool(budget=SearchBudget(1))

    first = tool.invoke({"query": "q"})
    second = tool.invoke({"query": "q"})

    assert "Result 1" in first
    assert second.startswith("ERROR:")
    assert "budget exhausted" in second
    # The second call must not reach the network.
    assert len(fake_httpx_get.calls) == 1


def test_tool_rate_limit_abort_returns_error_and_refunds_budget(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(json_data=make_searxng_payload(1))
    budget = SearchBudget(2)
    # An interval far larger than the timeout forces the second (concurrent)
    # slot to abort rather than sleep.
    tool = create_searxng_search_tool(
        timeout=1.0, min_request_interval=100.0, budget=budget
    )

    first = tool.invoke({"query": "q"})
    second = tool.invoke({"query": "q"})

    assert "Result 1" in first
    assert second.startswith("ERROR:")
    assert "rate limit" in second
    # Only the first search hit the network.
    assert len(fake_httpx_get.calls) == 1
    # The aborted search was refunded: one unit of budget is available again.
    assert budget.try_consume() is True
