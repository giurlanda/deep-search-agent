"""Smoke tests proving the Playwright browser setup works end to end."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def test_page_loads_successfully(page: Page, local_site: str) -> None:
    page.goto(f"{local_site}/sample_page.html")
    expect(page).to_have_title("Deep Search Agent — Sample Page")


def test_navigation_is_visible(page: Page, local_site: str) -> None:
    page.goto(f"{local_site}/sample_page.html")
    expect(page.get_by_role("navigation")).to_be_visible()
    expect(page.locator("#content")).to_contain_text("served locally")
