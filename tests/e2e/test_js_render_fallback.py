"""End-to-end tests for the headless-rendering fallback of ``fetch_url``.

Unlike the unit tests, these launch a real Chromium against a local JS-only
page, so they cover the Playwright code path itself.
"""

from __future__ import annotations

import pytest

from deep_search_agent.tools import create_fetch_url_tool

pytestmark = pytest.mark.e2e


def test_static_fetch_cannot_extract_a_js_only_page(local_site: str) -> None:
    tool = create_fetch_url_tool()

    output = tool.invoke({"url": f"{local_site}/js_only_page.html"})

    assert output.startswith("ERROR:")
    assert "no main content" in output


def test_js_render_fallback_recovers_a_js_only_page(local_site: str) -> None:
    tool = create_fetch_url_tool(enable_js_render_fallback=True)

    output = tool.invoke({"url": f"{local_site}/js_only_page.html"})

    assert not output.startswith("ERROR:")
    assert "injected by JavaScript" in output
