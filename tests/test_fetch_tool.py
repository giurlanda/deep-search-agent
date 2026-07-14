"""Unit tests for the fetch/extraction tool (no network access)."""

from __future__ import annotations

import httpx
import pytest

import deep_search_agent.tools.fetch as fetch_module
from deep_search_agent.tools import create_fetch_url_tool
from tests.conftest import FakeResponse

HTML_PAGE = """
<html>
  <head><title>Test article</title></head>
  <body>
    <nav>Home | About | Contact</nav>
    <article>
      <h1>Quantum computing breakthrough</h1>
      <p>Researchers announced a significant advance in error correction,
      demonstrating logical qubits with lower error rates than physical
      qubits for the first time in a scalable architecture.</p>
      <p>The result was replicated by two independent laboratories.</p>
    </article>
    <footer>Copyright 2026</footer>
  </body>
</html>
"""


def test_extracts_main_html_content(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(
        text=HTML_PAGE, headers={"content-type": "text/html; charset=utf-8"}
    )
    tool = create_fetch_url_tool()

    output = tool.invoke({"url": "https://example.org/article"})

    assert "error correction" in output
    assert not output.startswith("ERROR:")


def test_sends_browser_user_agent(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(
        text=HTML_PAGE, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool()

    tool.invoke({"url": "https://example.org/article"})

    _, kwargs = fake_httpx_get.calls[0]
    assert kwargs["headers"]["User-Agent"] in fetch_module.USER_AGENTS
    assert kwargs["follow_redirects"] is True


def test_truncates_long_content(fake_httpx_get):
    long_paragraph = "word " * 5000
    page = f"<html><body><article><p>{long_paragraph}</p></article></body></html>"
    fake_httpx_get.response = FakeResponse(
        text=page, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool(max_content_chars=100)

    output = tool.invoke({"url": "https://example.org/long"})

    assert output.endswith("...(content truncated)")
    assert len(output) < 200


def test_pdf_content_type_routes_to_pdf_extraction(fake_httpx_get, monkeypatch):
    class FakePage:
        @staticmethod
        def extract_text() -> str:
            return "Hello from the PDF"

    class FakeReader:
        def __init__(self, _stream) -> None:
            self.pages = [FakePage()]

    monkeypatch.setattr(fetch_module, "PdfReader", FakeReader)
    fake_httpx_get.response = FakeResponse(
        content=b"%PDF-1.4 fake", headers={"content-type": "application/pdf"}
    )
    tool = create_fetch_url_tool()

    output = tool.invoke({"url": "https://example.org/paper"})

    assert output == "Hello from the PDF"


def test_pdf_url_extension_routes_to_pdf_extraction(fake_httpx_get, monkeypatch):
    monkeypatch.setattr(
        fetch_module, "_extract_pdf_text", lambda content: "PDF via extension"
    )
    fake_httpx_get.response = FakeResponse(
        content=b"%PDF-1.4 fake", headers={"content-type": "application/octet-stream"}
    )
    tool = create_fetch_url_tool()

    output = tool.invoke({"url": "https://example.org/report.pdf?dl=1"})

    assert output == "PDF via extension"


def test_real_pdf_with_no_text_returns_error(fake_httpx_get):
    from io import BytesIO

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)

    fake_httpx_get.response = FakeResponse(
        content=buffer.getvalue(), headers={"content-type": "application/pdf"}
    )
    tool = create_fetch_url_tool()

    output = tool.invoke({"url": "https://example.org/blank.pdf"})

    assert output.startswith("ERROR:")
    assert "no extractable text" in output


def test_http_error_returns_error_string(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(status_code=403)
    tool = create_fetch_url_tool()

    output = tool.invoke({"url": "https://example.org/forbidden"})

    assert output.startswith("ERROR:")
    assert "403" in output


def test_network_error_returns_error_string(fake_httpx_get):
    fake_httpx_get.error = httpx.ConnectTimeout("timed out")
    tool = create_fetch_url_tool()

    output = tool.invoke({"url": "https://example.org/slow"})

    assert output.startswith("ERROR:")
    assert "could not fetch" in output


def test_unextractable_html_returns_error(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(
        text="<html><body><script>app()</script></body></html>",
        headers={"content-type": "text/html"},
    )
    tool = create_fetch_url_tool()

    output = tool.invoke({"url": "https://example.org/spa"})

    assert output.startswith("ERROR:")
    assert "no main content" in output


@pytest.mark.parametrize(
    ("url", "content_type", "expected"),
    [
        ("https://x.org/a.pdf", "text/html", True),
        ("https://x.org/a.PDF?x=1", "", True),
        ("https://x.org/a", "application/pdf", True),
        ("https://x.org/a.html", "text/html", False),
    ],
)
def test_is_pdf_detection(url, content_type, expected):
    assert fetch_module._is_pdf(url, content_type) is expected
