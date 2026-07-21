"""Unit tests for the fetch/extraction tool (no network access).

The headless-rendering fallback is exercised with Playwright stubbed out at
:func:`~deep_search_agent.tools.fetch._render_html`: the browser never launches,
but the thread isolation and the error handling around it stay real.
"""

from __future__ import annotations

import asyncio
import threading

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


# Static HTML of a JavaScript-only page: trafilatura finds nothing in it.
SPA_SHELL = "<html><body><div id='root'></div><script>app()</script></body></html>"


@pytest.fixture
def fake_render(monkeypatch):
    """Stub ``_render_html`` with a configurable, Playwright-free fake.

    Set ``holder.html`` (the rendered DOM) or ``holder.error`` (an exception to
    raise); ``holder.calls`` records the ``(url, timeout)`` of every call and
    ``holder.threads`` the thread each one ran on.
    """

    class Holder:
        html: str = ""
        error: Exception | None = None
        calls: list[tuple[str, float]]
        threads: list[str]

    holder = Holder()
    holder.calls = []
    holder.threads = []

    def _render(url: str, timeout: float) -> str:
        holder.calls.append((url, timeout))
        holder.threads.append(threading.current_thread().name)
        if holder.error is not None:
            raise holder.error
        return holder.html

    monkeypatch.setattr(fetch_module, "_render_html", _render)
    return holder


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


def test_truncates_long_content_keeping_head_and_tail(fake_httpx_get):
    paragraphs = "".join(f"<p>OPENING {'word ' * 20}</p>" for _ in range(15)) + "".join(
        f"<p>CLOSING {'word ' * 20}</p>" for _ in range(15)
    )
    page = f"<html><body><article>{paragraphs}</article></body></html>"
    fake_httpx_get.response = FakeResponse(
        text=page, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool(max_content_chars=2_000)

    output = tool.invoke({"url": "https://example.org/long"})

    assert fetch_module.TRUNCATION_MARKER in output
    assert output.startswith("OPENING")
    # The tail of the document survives — that is the point of the head+tail split.
    assert output.rstrip().endswith("word")
    assert "CLOSING" in output
    assert len(output) <= 2_000 + len(fetch_module.TRUNCATION_MARKER)


def test_short_content_is_not_truncated(fake_httpx_get):
    fake_httpx_get.response = FakeResponse(
        text=HTML_PAGE, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool(max_content_chars=20_000)

    output = tool.invoke({"url": "https://example.org/article"})

    assert fetch_module.TRUNCATION_MARKER not in output


def test_truncate_head_tail_snaps_to_paragraph_boundaries():
    # With max_chars=100 the head budget is 60 and the tail budget 40; both
    # paragraph breaks below sit in the outer half of their slice, so both
    # sides snap to them instead of cutting mid-paragraph.
    head = "A" * 40
    tail = "B" * 30
    text = "\n\n".join([head, "M" * 200, tail])

    output = fetch_module._truncate_head_tail(text, 100)

    part_head, part_tail = output.split(fetch_module.TRUNCATION_MARKER)
    assert part_head == head
    assert part_tail == tail


def test_truncate_head_tail_snaps_on_single_newlines():
    # trafilatura joins paragraphs with a single "\n", so that separator must
    # work as a cut point too.
    text = "\n".join(["A" * 40, "M" * 200, "B" * 30])

    output = fetch_module._truncate_head_tail(text, 100)

    part_head, part_tail = output.split(fetch_module.TRUNCATION_MARKER)
    assert part_head == "A" * 40
    assert part_tail == "B" * 30


def test_truncate_head_tail_without_paragraphs_falls_back_to_hard_cut():
    text = "x" * 500

    output = fetch_module._truncate_head_tail(text, 100)

    part_head, part_tail = output.split(fetch_module.TRUNCATION_MARKER)
    assert len(part_head) == 60
    assert len(part_tail) == 40


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
        text=SPA_SHELL,
        headers={"content-type": "text/html"},
    )
    tool = create_fetch_url_tool()

    output = tool.invoke({"url": "https://example.org/spa"})

    assert output.startswith("ERROR:")
    assert "no main content" in output


def test_js_fallback_is_off_by_default(fake_httpx_get, fake_render):
    fake_render.html = HTML_PAGE
    fake_httpx_get.response = FakeResponse(
        text=SPA_SHELL, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool()

    output = tool.invoke({"url": "https://example.org/spa"})

    assert output.startswith("ERROR:")
    assert fake_render.calls == []


def test_js_fallback_recovers_a_javascript_only_page(fake_httpx_get, fake_render):
    fake_render.html = HTML_PAGE
    fake_httpx_get.response = FakeResponse(
        text=SPA_SHELL, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool(enable_js_render_fallback=True, js_render_timeout=12.0)

    output = tool.invoke({"url": "https://example.org/spa"})

    assert "error correction" in output
    assert not output.startswith("ERROR:")
    assert fake_render.calls == [("https://example.org/spa", 12.0)]


def test_js_fallback_is_skipped_when_static_extraction_succeeds(
    fake_httpx_get, fake_render
):
    fake_httpx_get.response = FakeResponse(
        text=HTML_PAGE, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool(enable_js_render_fallback=True)

    output = tool.invoke({"url": "https://example.org/article"})

    assert "error correction" in output
    assert fake_render.calls == []


def test_js_fallback_renders_off_the_calling_thread(fake_httpx_get, fake_render):
    # Playwright's sync API refuses to run on a thread owning a live asyncio
    # loop, which is exactly what an async caller hands the tool.
    fake_render.html = HTML_PAGE
    fake_httpx_get.response = FakeResponse(
        text=SPA_SHELL, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool(enable_js_render_fallback=True)

    async def call() -> str:
        return await tool.ainvoke({"url": "https://example.org/spa"})

    output = asyncio.run(call())

    assert "error correction" in output
    assert fake_render.threads[0] != threading.current_thread().name


def test_js_fallback_on_still_empty_page_returns_error(fake_httpx_get, fake_render):
    fake_render.html = "<html><body><div id='root'></div></body></html>"
    fake_httpx_get.response = FakeResponse(
        text=SPA_SHELL, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool(enable_js_render_fallback=True)

    output = tool.invoke({"url": "https://example.org/spa"})

    assert output.startswith("ERROR:")
    assert "no main content" in output
    assert len(fake_render.calls) == 1


def test_js_fallback_without_playwright_returns_actionable_error(
    fake_httpx_get, fake_render
):
    fake_render.error = ImportError("No module named 'playwright'")
    fake_httpx_get.response = FakeResponse(
        text=SPA_SHELL, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool(enable_js_render_fallback=True)

    output = tool.invoke({"url": "https://example.org/spa"})

    assert output.startswith("ERROR:")
    assert "js-render" in output


def test_js_fallback_render_failure_returns_error_string(fake_httpx_get, fake_render):
    fake_render.error = RuntimeError("browser executable not found")
    fake_httpx_get.response = FakeResponse(
        text=SPA_SHELL, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool(enable_js_render_fallback=True)

    output = tool.invoke({"url": "https://example.org/spa"})

    assert output.startswith("ERROR:")
    assert "browser executable not found" in output


def test_js_fallback_output_is_truncated_like_static_content(
    fake_httpx_get, fake_render
):
    paragraphs = "".join(f"<p>{'word ' * 40}</p>" for _ in range(30))
    fake_render.html = f"<html><body><article>{paragraphs}</article></body></html>"
    fake_httpx_get.response = FakeResponse(
        text=SPA_SHELL, headers={"content-type": "text/html"}
    )
    tool = create_fetch_url_tool(
        enable_js_render_fallback=True, max_content_chars=1_000
    )

    output = tool.invoke({"url": "https://example.org/spa"})

    assert fetch_module.TRUNCATION_MARKER in output
    assert len(output) <= 1_000 + len(fetch_module.TRUNCATION_MARKER)


def test_js_fallback_does_not_apply_to_pdfs(fake_httpx_get, fake_render, monkeypatch):
    monkeypatch.setattr(fetch_module, "_extract_pdf_text", lambda content: "")
    fake_httpx_get.response = FakeResponse(
        content=b"%PDF-1.4 fake", headers={"content-type": "application/pdf"}
    )
    tool = create_fetch_url_tool(enable_js_render_fallback=True)

    output = tool.invoke({"url": "https://example.org/scan.pdf"})

    assert "no extractable text" in output
    assert fake_render.calls == []


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
