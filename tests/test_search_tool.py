"""Unit tests for the SearxNG search tool (no network access)."""

from __future__ import annotations

import httpx

from deep_search_agent.tools import create_searxng_search_tool
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
