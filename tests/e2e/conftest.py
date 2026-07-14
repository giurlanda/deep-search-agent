"""Fixtures for browser-based e2e tests (pytest-playwright).

These tests exercise real browser rendering against a local static server,
so they need no network access and stay deterministic.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def local_site() -> Iterator[str]:
    """Serve ``tests/e2e/test_data/`` on an ephemeral port; yield its base URL."""
    handler = partial(SimpleHTTPRequestHandler, directory=str(TEST_DATA_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
