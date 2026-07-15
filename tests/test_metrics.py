"""Unit tests for SessionMetrics and its collecting middleware."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from deep_search_agent import CycleMetrics, SessionMetrics, SubagentStats
from deep_search_agent.metrics import (
    _OrchestratorMetricsMiddleware,
    _SubagentMetricsMiddleware,
)


class FakeToolCall:
    """Minimal stand-in for a ``ToolCallRequest`` exposing ``tool_call``."""

    def __init__(self, name: str, args: dict | None = None) -> None:
        self.tool_call = {"name": name, "args": args or {}}


def _handler(_request: Any) -> str:
    """A tool handler that just echoes a constant (result is opaque here)."""
    return "ok"


# --- SessionMetrics recording -------------------------------------------------


def test_orchestrator_tool_counts_per_cycle_and_global():
    m = SessionMetrics()
    m._on_run_start()

    m._record_orchestrator_tool("write_todos")
    m._record_orchestrator_tool("write_file")
    m._record_orchestrator_tool("write_file")

    assert m.global_tool_calls == {"write_todos": 1, "write_file": 2}
    assert m.cycles[0].orchestrator_tool_calls == {"write_todos": 1, "write_file": 2}


def test_task_tool_is_not_counted():
    m = SessionMetrics()
    m._on_run_start()

    m._record_orchestrator_tool("task")
    m._record_orchestrator_tool("write_todos")

    assert m.global_tool_calls == {"write_todos": 1}
    assert "task" not in m.cycles[0].orchestrator_tool_calls


def test_cycle_boundary_advances_index():
    m = SessionMetrics()
    m._on_run_start()

    m._record_orchestrator_tool("write_todos")
    m._on_cycle_end()  # close cycle 0
    m._record_orchestrator_tool("write_file")

    assert m.cycle_count == 2
    assert m.cycles[0].orchestrator_tool_calls == {"write_todos": 1}
    assert m.cycles[1].orchestrator_tool_calls == {"write_file": 1}


def test_subagent_invocation_counts_and_timings():
    m = SessionMetrics()
    m._on_run_start()

    m._record_subagent_invocation("search-agent", 2.0)
    m._record_subagent_invocation("search-agent", 4.0)
    m._record_subagent_invocation("fetch-agent", 1.0)

    assert m.global_subagent_invocations == {"search-agent": 2, "fetch-agent": 1}
    stats = m.subagent_stats
    assert stats["search-agent"] == SubagentStats(
        count=2, total_time=6.0, min_time=2.0, max_time=4.0
    )
    assert stats["search-agent"].avg_time == 3.0
    assert stats["fetch-agent"].avg_time == 1.0
    # Per-cycle invocation counts are attributed to the current cycle.
    assert m.cycles[0].subagent_invocations == {"search-agent": 2, "fetch-agent": 1}


def test_subagent_tool_calls_feed_global_and_cycle():
    m = SessionMetrics()
    m._on_run_start()

    m._record_orchestrator_tool("write_file")
    m._record_subagent_tool("search-agent", "internet_search")
    m._record_subagent_tool("search-agent", "internet_search")
    m._record_subagent_tool("fetch-agent", "fetch_url")

    # Global tool counts combine orchestrator and sub-agent tools.
    assert m.global_tool_calls == {
        "write_file": 1,
        "internet_search": 2,
        "fetch_url": 1,
    }
    assert m.cycles[0].subagent_tool_calls == {
        "search-agent": {"internet_search": 2},
        "fetch-agent": {"fetch_url": 1},
    }


def test_stats_absent_before_invocation():
    m = SessionMetrics()
    assert m.subagent_stats == {}
    assert m.cycle_count == 0
    assert m.total_duration == 0.0


def test_to_dict_is_serializable_snapshot():
    m = SessionMetrics()
    m._on_run_start()
    m._record_orchestrator_tool("write_todos")
    m._record_subagent_invocation("search-agent", 3.0)
    m._record_subagent_tool("search-agent", "internet_search")
    m._on_cycle_end()

    snap = m.to_dict()
    assert snap["tool_calls"] == {"write_todos": 1, "internet_search": 1}
    assert snap["subagent_invocations"] == {"search-agent": 1}
    assert snap["subagent_stats"]["search-agent"] == {
        "count": 1,
        "total_time": 3.0,
        "avg_time": 3.0,
        "min_time": 3.0,
        "max_time": 3.0,
    }
    assert snap["cycles"][0]["orchestrator_tool_calls"] == {"write_todos": 1}
    assert snap["cycles"][0]["subagent_tool_calls"] == {
        "search-agent": {"internet_search": 1}
    }
    # snapshot is an alias of to_dict.
    assert m.snapshot() == snap


def test_reset_clears_everything():
    m = SessionMetrics()
    m._on_run_start()
    m._record_orchestrator_tool("write_todos")
    m._record_subagent_invocation("search-agent", 1.0)
    m._on_cycle_end()

    m.reset()

    assert m.cycle_count == 0
    assert m.global_tool_calls == {}
    assert m.global_subagent_invocations == {}
    assert m.subagent_stats == {}
    assert m.total_duration == 0.0
    # After reset the cycle index restarts from 0.
    m._on_run_start()
    m._record_orchestrator_tool("ls")
    assert m.cycles[0].orchestrator_tool_calls == {"ls": 1}


def test_returned_structures_are_copies():
    m = SessionMetrics()
    m._on_run_start()
    m._record_orchestrator_tool("write_todos")

    calls = m.global_tool_calls
    calls["write_todos"] = 999
    assert m.global_tool_calls == {"write_todos": 1}

    cycles = m.cycles
    assert isinstance(cycles[0], CycleMetrics)
    cycles[0].orchestrator_tool_calls["write_todos"] = 999
    assert m.cycles[0].orchestrator_tool_calls == {"write_todos": 1}


def test_total_duration_accumulates_across_runs():
    m = SessionMetrics()
    m._on_run_start()
    m._on_cycle_end()
    first = m.total_duration
    assert first >= 0.0

    m._on_run_start()
    m._on_cycle_end()
    assert m.total_duration >= first


def test_recording_is_thread_safe():
    m = SessionMetrics()
    m._on_run_start()

    def worker() -> None:
        for _ in range(1000):
            m._record_orchestrator_tool("write_file")
            m._record_subagent_tool("search-agent", "internet_search")
            m._record_subagent_invocation("search-agent", 0.5)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert m.global_tool_calls["write_file"] == 8000
    assert m.global_tool_calls["internet_search"] == 8000
    assert m.subagent_stats["search-agent"].count == 8000


# --- Orchestrator middleware --------------------------------------------------


def test_orchestrator_middleware_counts_and_delimits_cycles():
    m = SessionMetrics()
    mw = _OrchestratorMetricsMiddleware(m)

    mw.before_agent({}, None)
    mw.wrap_tool_call(FakeToolCall("write_todos"), _handler)
    mw.after_agent({}, None)  # cycle boundary

    assert m.global_tool_calls == {"write_todos": 1}
    # Only cycle 0 has activity; the boundary advances the index without
    # creating a phantom empty cycle 1.
    assert m.cycle_count == 1
    assert m.cycles[0].orchestrator_tool_calls == {"write_todos": 1}
    assert m.total_duration >= 0.0


def test_orchestrator_middleware_times_and_counts_subagent_via_task():
    m = SessionMetrics()
    mw = _OrchestratorMetricsMiddleware(m)
    mw.before_agent({}, None)

    request = FakeToolCall("task", {"subagent_type": "fetch-agent", "description": "x"})
    mw.wrap_tool_call(request, _handler)

    # The task call is recorded as a sub-agent invocation, not an orchestrator
    # tool.
    assert m.global_tool_calls == {}
    assert m.global_subagent_invocations == {"fetch-agent": 1}
    assert m.subagent_stats["fetch-agent"].count == 1
    assert m.subagent_stats["fetch-agent"].min_time is not None
    assert m.cycles[0].subagent_invocations == {"fetch-agent": 1}


def test_orchestrator_middleware_ignores_task_without_subagent_type():
    m = SessionMetrics()
    mw = _OrchestratorMetricsMiddleware(m)
    mw.before_agent({}, None)

    mw.wrap_tool_call(FakeToolCall("task"), _handler)  # malformed: no subagent_type

    assert m.global_tool_calls == {}
    assert m.global_subagent_invocations == {}


def test_orchestrator_middleware_wrap_returns_handler_result():
    m = SessionMetrics()
    mw = _OrchestratorMetricsMiddleware(m)
    sentinel = object()

    result = mw.wrap_tool_call(FakeToolCall("write_file"), lambda _r: sentinel)

    assert result is sentinel


def test_orchestrator_middleware_async_hooks():
    m = SessionMetrics()
    mw = _OrchestratorMetricsMiddleware(m)

    async def handler(_request: Any) -> str:
        return "ok"

    asyncio.run(mw.abefore_agent({}, None))
    asyncio.run(mw.awrap_tool_call(FakeToolCall("write_file"), handler))
    asyncio.run(
        mw.awrap_tool_call(
            FakeToolCall("task", {"subagent_type": "search-agent"}), handler
        )
    )
    asyncio.run(mw.aafter_agent({}, None))

    assert m.global_tool_calls == {"write_file": 1}
    assert m.global_subagent_invocations == {"search-agent": 1}
    assert m.cycle_count == 1


# --- Sub-agent middleware -----------------------------------------------------


def test_subagent_middleware_counts_tool_calls():
    m = SessionMetrics()
    m._on_run_start()
    mw = _SubagentMetricsMiddleware(m, "search-agent")

    mw.wrap_tool_call(FakeToolCall("internet_search"), _handler)
    mw.wrap_tool_call(FakeToolCall("internet_search"), _handler)

    assert m.cycles[0].subagent_tool_calls == {"search-agent": {"internet_search": 2}}
    # Sub-agent tool calls also feed the global tool totals.
    assert m.global_tool_calls == {"internet_search": 2}


def test_subagent_middleware_async_hook():
    m = SessionMetrics()
    m._on_run_start()
    mw = _SubagentMetricsMiddleware(m, "fetch-agent")

    async def handler(_request: Any) -> str:
        return "ok"

    asyncio.run(mw.awrap_tool_call(FakeToolCall("fetch_url"), handler))

    assert m.cycles[0].subagent_tool_calls == {"fetch-agent": {"fetch_url": 1}}


def test_subagent_middleware_wrap_returns_handler_result():
    m = SessionMetrics()
    m._on_run_start()
    mw = _SubagentMetricsMiddleware(m, "fetch-agent")
    sentinel = object()

    result = mw.wrap_tool_call(FakeToolCall("fetch_url"), lambda _r: sentinel)

    assert result is sentinel
