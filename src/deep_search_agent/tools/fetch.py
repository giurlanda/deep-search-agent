"""URL fetching and content-extraction tool.

Exposes :func:`create_fetch_url_tool`, a factory that builds a LangChain tool
which downloads a URL with realistic browser headers and extracts its main
content:

- HTML pages are cleaned with ``trafilatura`` (boilerplate removal).
- PDF documents (detected via ``Content-Type`` or a ``.pdf`` extension) are
  read with ``pypdf``.

Failures are returned as ``ERROR: ...`` strings instead of raising, so the
calling agent can reroute to a different source without crashing the flow.
"""

from __future__ import annotations

import random
from io import BytesIO

import httpx
import trafilatura
from langchain_core.tools import BaseTool, tool
from pypdf import PdfReader

# Realistic desktop browser User-Agent strings, rotated per request to reduce
# the chance of being served bot-detection pages.
USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 "
    "Firefox/127.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
)


def _build_headers() -> dict[str, str]:
    """Build request headers imitating a real browser, with a random UA."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "application/pdf,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9,it;q=0.8",
    }


def _extract_pdf_text(content: bytes) -> str:
    """Extract plain text from PDF bytes, page by page.

    Args:
        content: Raw PDF document bytes.

    Returns:
        The concatenated text of all pages (pages with no extractable text
        are skipped).
    """
    reader = PdfReader(BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(text for text in pages if text.strip())


def _is_pdf(url: str, content_type: str) -> bool:
    """Return True when the response should be treated as a PDF document."""
    return "application/pdf" in content_type or url.lower().split("?")[0].endswith(
        ".pdf"
    )


def create_fetch_url_tool(
    *,
    timeout: float = 20.0,
    max_content_chars: int = 20_000,
) -> BaseTool:
    """Build a tool that downloads a URL and extracts its main content.

    Args:
        timeout: Per-request timeout in seconds.
        max_content_chars: Maximum number of characters returned; longer
            content is truncated with an explicit marker.

    Returns:
        A LangChain tool named ``fetch_url`` that takes a ``url`` string and
        returns the extracted text (HTML cleaned via trafilatura, PDFs read
        via pypdf), or an ``ERROR: ...`` string on failure.
    """

    @tool
    def fetch_url(url: str) -> str:
        """Download a web page or PDF and return its cleaned main content.

        Args:
            url: The absolute URL to fetch. Both HTML pages and PDF
                documents are supported.

        Returns:
            The extracted text content (truncated if very long), or an
            ``ERROR: ...`` message if the download/extraction failed.
        """
        try:
            response = httpx.get(
                url,
                headers=_build_headers(),
                timeout=timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return (
                f"ERROR: HTTP {exc.response.status_code} while fetching {url}. "
                "The page may be protected or gone; try another source."
            )
        except httpx.HTTPError as exc:
            return f"ERROR: could not fetch {url}: {exc}. Try another source."

        content_type = response.headers.get("content-type", "")
        if _is_pdf(url, content_type):
            try:
                text = _extract_pdf_text(response.content)
            except Exception as exc:  # pypdf raises many exception types
                return f"ERROR: could not parse PDF at {url}: {exc}."
            if not text.strip():
                return (
                    f"ERROR: the PDF at {url} contains no extractable text "
                    "(it may be a scanned document)."
                )
        else:
            text = (
                trafilatura.extract(
                    response.text,
                    url=url,
                    include_comments=False,
                    include_tables=True,
                )
                or ""
            )
            if not text.strip():
                return (
                    f"ERROR: no main content could be extracted from {url} "
                    "(possibly a JavaScript-only page or a bot wall)."
                )

        if len(text) > max_content_chars:
            text = text[:max_content_chars] + "\n\n...(content truncated)"
        return text

    return fetch_url
