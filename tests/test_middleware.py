"""Unit tests for the deep-search middleware."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_search_agent import (
    DEEP_SEARCH_RUBRIC,
    DeepSearchRubricMiddleware,
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


# --- DeepSearchRubricMiddleware ----------------------------------------------

# Longer than deepagents' 4,000-char per-message cap, with a unique marker at
# the very end (where our reports place the bibliography — issue #22).
_LONG_ANSWER = "A" * 5_000 + "\n[1] https://example.com/final-source END_OF_REPORT"


def _payload(messages):
    mw = DeepSearchRubricMiddleware(model="fake")  # lazy grader: never built
    return mw._build_grader_payload({"rubric": "- criterion", "messages": messages}, 0)


def test_final_answer_passed_untruncated():
    payload = _payload([HumanMessage(content="Q"), AIMessage(content=_LONG_ANSWER)])

    # The whole final answer — including its tail — reaches the grader.
    assert _LONG_ANSWER in payload
    assert "END_OF_REPORT" in payload


def test_earlier_messages_still_truncated():
    tool_tail = "TOOL_TAIL_MARKER"
    messages = [
        HumanMessage(content="Q"),
        ToolMessage(content="T" * 5_000 + tool_tail, tool_call_id="t1", name="search"),
        AIMessage(content=_LONG_ANSWER),
    ]

    payload = _payload(messages)

    # The oversized intermediate tool output is still bounded...
    assert tool_tail not in payload
    assert "...(truncated)" in payload
    # ...while the final answer is not.
    assert "END_OF_REPORT" in payload


def test_payload_preserves_rubric_and_delimiters():
    payload = _payload([HumanMessage(content="Q"), AIMessage(content=_LONG_ANSWER)])

    assert "- criterion" in payload
    assert "<rubric-" in payload
    assert "<transcript-" in payload


def test_single_final_message_untruncated():
    payload = _payload([AIMessage(content=_LONG_ANSWER)])

    assert "END_OF_REPORT" in payload


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
