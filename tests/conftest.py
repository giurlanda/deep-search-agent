"""Shared test fixtures for the deep_search_agent test suite."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest


class FakeResponse:
    """Minimal stand-in for ``httpx.Response`` used by the tool tests."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: Any | None = None,
        content: bytes = b"",
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.content = content
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:
        if self._json_data is None:
            raise ValueError("no JSON")
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )


@pytest.fixture
def fake_httpx_get(monkeypatch):
    """Patch ``httpx.get`` with a stub whose response is configurable.

    Returns a holder object: set ``holder.response`` (a FakeResponse) or
    ``holder.error`` (an exception to raise); ``holder.calls`` records the
    (url, kwargs) of every call.
    """

    class Holder:
        response: FakeResponse | None = None
        error: Exception | None = None
        calls: list[tuple[str, dict]] = []

    holder = Holder()

    def _get(url: str, **kwargs: Any) -> FakeResponse:
        holder.calls.append((url, kwargs))
        if holder.error is not None:
            raise holder.error
        assert holder.response is not None, "set holder.response in the test"
        return holder.response

    monkeypatch.setattr(httpx, "get", _get)
    return holder


def make_searxng_payload(n: int) -> dict:
    """Build a SearxNG-like JSON payload with ``n`` results."""
    return {
        "results": [
            {
                "title": f"Result {i}",
                "url": f"https://example.org/{i}",
                "content": f"Snippet {i}",
                "engine": "duckduckgo",
                "publishedDate": "2026-01-01",
            }
            for i in range(1, n + 1)
        ]
    }


def searxng_json(n: int) -> str:
    """JSON string form of :func:`make_searxng_payload` (for convenience)."""
    return json.dumps(make_searxng_payload(n))
