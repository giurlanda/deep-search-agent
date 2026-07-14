"""Unit tests for DefaultRubricMiddleware."""

from __future__ import annotations

import pytest

from deep_search_agent import DEEP_SEARCH_RUBRIC, DefaultRubricMiddleware


def test_injects_rubric_when_absent():
    mw = DefaultRubricMiddleware(DEEP_SEARCH_RUBRIC)

    update = mw.before_agent({}, None)

    assert update == {"rubric": DEEP_SEARCH_RUBRIC}


def test_injects_rubric_when_empty():
    mw = DefaultRubricMiddleware("- criterion")

    update = mw.before_agent({"rubric": ""}, None)

    assert update == {"rubric": "- criterion"}


def test_preserves_caller_rubric():
    mw = DefaultRubricMiddleware(DEEP_SEARCH_RUBRIC)

    update = mw.before_agent({"rubric": "- custom criterion"}, None)

    assert update is None


def test_async_hook_matches_sync():
    import asyncio

    mw = DefaultRubricMiddleware("- criterion")

    assert asyncio.run(mw.abefore_agent({}, None)) == {"rubric": "- criterion"}
    assert asyncio.run(mw.abefore_agent({"rubric": "x"}, None)) is None


def test_rejects_empty_rubric():
    with pytest.raises(ValueError, match="non-empty rubric"):
        DefaultRubricMiddleware("   ")
