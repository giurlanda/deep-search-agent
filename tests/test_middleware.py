"""Unit tests for DefaultRubricMiddleware and SearchBudgetResetMiddleware."""

from __future__ import annotations

import pytest

from deep_search_agent import (
    DEEP_SEARCH_RUBRIC,
    DefaultRubricMiddleware,
    SearchBudget,
    SearchBudgetResetMiddleware,
)


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


# --- SearchBudgetResetMiddleware ---------------------------------------------


def test_before_agent_resets_budget():
    budget = SearchBudget(1)
    assert budget.try_consume() is True
    assert budget.try_consume() is False
    mw = SearchBudgetResetMiddleware(budget)

    mw.before_agent({}, None)

    assert budget.try_consume() is True


def test_after_agent_resets_budget_for_next_cycle():
    budget = SearchBudget(1)
    assert budget.try_consume() is True
    assert budget.try_consume() is False
    mw = SearchBudgetResetMiddleware(budget)

    mw.after_agent({}, None)

    assert budget.try_consume() is True


def test_async_hooks_reset_budget():
    import asyncio

    budget = SearchBudget(1)
    assert budget.try_consume() is True
    mw = SearchBudgetResetMiddleware(budget)

    asyncio.run(mw.abefore_agent({}, None))
    assert budget.try_consume() is True

    budget.try_consume()  # exhaust again
    asyncio.run(mw.aafter_agent({}, None))
    assert budget.try_consume() is True
