"""Session-level observability metrics for the deep-search agent.

:class:`SessionMetrics` is a thread-safe collector that aggregates, over a whole
deep-search session, how much work the orchestrator and its sub-agents did. A
session spans the lifetime of the object: metrics accumulate across every
research (iteration) cycle and across successive invocations of a reused agent
graph until :meth:`SessionMetrics.reset` is called.

It is wired into an agent by passing an instance to
:func:`~deep_search_agent.factory.create_deep_search_agent` via its ``metrics``
parameter. The factory injects two observation-only middleware:

- :class:`_OrchestratorMetricsMiddleware` on the orchestrator: it times the run
  (overall execution time), delimits research cycles at every ``after_agent``
  boundary (where ``RubricMiddleware`` loops back for another cycle), counts the
  orchestrator's own tool calls, and — because the sub-agent delegation tool
  (``task``) runs the whole sub-agent synchronously inside the tool call — times
  and counts each sub-agent invocation by wrapping that call. The ``task`` tool
  itself is *not* counted as an ordinary orchestrator tool.
- :class:`_SubagentMetricsMiddleware`, one instance per built-in sub-agent
  (``search-agent``, ``fetch-agent``, ``fact-check-agent``): it counts the tool
  calls the sub-agent makes, attributing them to the research cycle in progress
  on the orchestrator.

Timing and invocation counting live at the orchestrator's ``task`` boundary
rather than in the sub-agent's own ``before_agent``/``after_agent`` on purpose:
LangGraph runs those two hooks in separate node contexts, so a start timestamp
stashed in one is not visible in the other. Wrapping the ``task`` tool call
brackets the entire sub-agent run in a single context, which is correct under
both threads and asyncio tasks. Because deepagents may run sub-agents
concurrently, every mutation is serialized by a re-entrant lock.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from langchain_core.messages import ToolMessage
    from langgraph.runtime import Runtime
    from langgraph.types import Command

# Delegation to a sub-agent goes through this single built-in orchestrator tool
# (deepagents' ``task`` tool, dispatched by ``subagent_type``). It is excluded
# from the orchestrator tool-call counts because sub-agent invocations are
# tracked on their own, richer axis (invocation counts + timings + tool calls).
_TASK_TOOL_NAME = "task"
_SUBAGENT_TYPE_ARG = "subagent_type"


@dataclass(frozen=True)
class SubagentStats:
    """Aggregated execution stats for a single sub-agent over the session.

    Attributes:
        count: Number of completed invocations of the sub-agent.
        total_time: Sum of the wall-clock durations of those invocations, in
            seconds.
        min_time: Shortest invocation duration in seconds, or ``None`` if the
            sub-agent was never invoked.
        max_time: Longest invocation duration in seconds, or ``None`` if the
            sub-agent was never invoked.
    """

    count: int
    total_time: float
    min_time: float | None
    max_time: float | None

    @property
    def avg_time(self) -> float | None:
        """Mean invocation duration in seconds, or ``None`` if never invoked."""
        if self.count == 0:
            return None
        return self.total_time / self.count


@dataclass(frozen=True)
class CycleMetrics:
    """Per-cycle counters for one research (iteration) cycle.

    Attributes:
        orchestrator_tool_calls: How many times the orchestrator invoked each of
            its own tools during the cycle (the ``task`` delegation tool is
            excluded).
        subagent_invocations: How many times each sub-agent was invoked during
            the cycle.
        subagent_tool_calls: For each sub-agent, how many times it invoked each
            of its tools during the cycle.
    """

    orchestrator_tool_calls: dict[str, int]
    subagent_invocations: dict[str, int]
    subagent_tool_calls: dict[str, dict[str, int]]


class _CycleData:
    """Mutable accumulator backing a single :class:`CycleMetrics` snapshot."""

    __slots__ = (
        "orchestrator_tool_calls",
        "subagent_invocations",
        "subagent_tool_calls",
    )

    def __init__(self) -> None:
        self.orchestrator_tool_calls: Counter[str] = Counter()
        self.subagent_invocations: Counter[str] = Counter()
        self.subagent_tool_calls: defaultdict[str, Counter[str]] = defaultdict(Counter)

    def snapshot(self) -> CycleMetrics:
        return CycleMetrics(
            orchestrator_tool_calls=dict(self.orchestrator_tool_calls),
            subagent_invocations=dict(self.subagent_invocations),
            subagent_tool_calls={
                name: dict(counts) for name, counts in self.subagent_tool_calls.items()
            },
        )


class SessionMetrics:
    """Thread-safe collector of per-cycle and global deep-search metrics.

    Pass an instance to
    :func:`~deep_search_agent.factory.create_deep_search_agent` through its
    ``metrics`` parameter, run the agent, then read the results either through
    the typed properties (:attr:`cycles`, :attr:`global_tool_calls`,
    :attr:`subagent_stats`, ...) or as a plain JSON-serializable mapping via
    :meth:`to_dict`. Metrics accumulate for the lifetime of the object; call
    :meth:`reset` to start a fresh session.

    All reads return copies, so the returned structures never mutate under the
    caller while the agent keeps running.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._global_tool_calls: Counter[str] = Counter()
        self._global_subagent_invocations: Counter[str] = Counter()
        self._subagent_count: Counter[str] = Counter()
        self._subagent_total_time: defaultdict[str, float] = defaultdict(float)
        self._subagent_min_time: dict[str, float] = {}
        self._subagent_max_time: dict[str, float] = {}
        self._cycles: list[_CycleData] = []
        self._current_cycle = 0
        self._total_duration = 0.0
        self._run_start: float | None = None
        self._run_baseline = 0.0

    # -- internal recording API (called by the metrics middleware) ------------

    def _ensure_cycle(self, index: int) -> _CycleData:
        """Return the cycle accumulator for ``index``, creating it if needed."""
        while len(self._cycles) <= index:
            self._cycles.append(_CycleData())
        return self._cycles[index]

    def _on_run_start(self) -> None:
        """Mark the start of an orchestrator invocation (for overall timing)."""
        with self._lock:
            self._run_baseline = self._total_duration
            self._run_start = time.monotonic()
            self._ensure_cycle(self._current_cycle)

    def _on_cycle_end(self) -> None:
        """Close the current research cycle and advance the cycle index.

        Fires once per ``RubricMiddleware`` iteration (the orchestrator's
        ``after_agent`` boundary). The overall duration is recomputed from the
        run baseline on every call, so the final cycle's timestamp yields the
        true end of the invocation.
        """
        with self._lock:
            if self._run_start is not None:
                self._total_duration = self._run_baseline + (
                    time.monotonic() - self._run_start
                )
            self._current_cycle += 1

    def _record_orchestrator_tool(self, tool_name: str) -> None:
        """Count one orchestrator tool call (the ``task`` tool is ignored)."""
        if tool_name == _TASK_TOOL_NAME:
            return
        with self._lock:
            cycle = self._ensure_cycle(self._current_cycle)
            cycle.orchestrator_tool_calls[tool_name] += 1
            self._global_tool_calls[tool_name] += 1

    def _record_subagent_invocation(self, subagent: str, duration: float) -> None:
        """Count one completed sub-agent invocation and record its duration."""
        with self._lock:
            self._subagent_count[subagent] += 1
            self._global_subagent_invocations[subagent] += 1
            self._subagent_total_time[subagent] += duration
            prev_min = self._subagent_min_time.get(subagent)
            if prev_min is None or duration < prev_min:
                self._subagent_min_time[subagent] = duration
            prev_max = self._subagent_max_time.get(subagent)
            if prev_max is None or duration > prev_max:
                self._subagent_max_time[subagent] = duration
            cycle = self._ensure_cycle(self._current_cycle)
            cycle.subagent_invocations[subagent] += 1

    def _record_subagent_tool(self, subagent: str, tool_name: str) -> None:
        """Count one tool call made by ``subagent`` in the current cycle."""
        with self._lock:
            cycle = self._ensure_cycle(self._current_cycle)
            cycle.subagent_tool_calls[subagent][tool_name] += 1
            self._global_tool_calls[tool_name] += 1

    # -- public read API ------------------------------------------------------

    @property
    def total_duration(self) -> float:
        """Overall orchestrator execution time (seconds), summed over runs."""
        with self._lock:
            return self._total_duration

    @property
    def cycle_count(self) -> int:
        """Number of research cycles recorded so far in the session."""
        with self._lock:
            return len(self._cycles)

    @property
    def cycles(self) -> tuple[CycleMetrics, ...]:
        """Immutable per-cycle snapshots, oldest first."""
        with self._lock:
            return tuple(cycle.snapshot() for cycle in self._cycles)

    @property
    def global_tool_calls(self) -> dict[str, int]:
        """Total call count per tool across the orchestrator and sub-agents."""
        with self._lock:
            return dict(self._global_tool_calls)

    @property
    def global_subagent_invocations(self) -> dict[str, int]:
        """Total invocation count per sub-agent across the session."""
        with self._lock:
            return dict(self._global_subagent_invocations)

    @property
    def subagent_stats(self) -> dict[str, SubagentStats]:
        """Per-sub-agent execution stats (count and avg/min/max duration)."""
        with self._lock:
            return {
                name: SubagentStats(
                    count=count,
                    total_time=self._subagent_total_time[name],
                    min_time=self._subagent_min_time.get(name),
                    max_time=self._subagent_max_time.get(name),
                )
                for name, count in self._subagent_count.items()
            }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of every collected metric.

        Returns:
            A nested mapping with ``total_duration``, global ``tool_calls`` and
            ``subagent_invocations``, per-sub-agent ``subagent_stats`` (each with
            ``count``/``total_time``/``avg_time``/``min_time``/``max_time``), and
            a ``cycles`` list of per-cycle counters.
        """
        with self._lock:
            return {
                "total_duration": self._total_duration,
                "tool_calls": dict(self._global_tool_calls),
                "subagent_invocations": dict(self._global_subagent_invocations),
                "subagent_stats": {
                    name: {
                        "count": count,
                        "total_time": self._subagent_total_time[name],
                        "avg_time": self._subagent_total_time[name] / count
                        if count
                        else None,
                        "min_time": self._subagent_min_time.get(name),
                        "max_time": self._subagent_max_time.get(name),
                    }
                    for name, count in self._subagent_count.items()
                },
                "cycles": [
                    {
                        "orchestrator_tool_calls": dict(cycle.orchestrator_tool_calls),
                        "subagent_invocations": dict(cycle.subagent_invocations),
                        "subagent_tool_calls": {
                            name: dict(counts)
                            for name, counts in cycle.subagent_tool_calls.items()
                        },
                    }
                    for cycle in self._cycles
                ],
            }

    # ``snapshot`` reads as a nicer verb at call sites; keep it as an alias.
    snapshot = to_dict

    def reset(self) -> None:
        """Clear every collected metric and start a fresh session."""
        with self._lock:
            self._global_tool_calls.clear()
            self._global_subagent_invocations.clear()
            self._subagent_count.clear()
            self._subagent_total_time.clear()
            self._subagent_min_time.clear()
            self._subagent_max_time.clear()
            self._cycles.clear()
            self._current_cycle = 0
            self._total_duration = 0.0
            self._run_start = None
            self._run_baseline = 0.0


class _OrchestratorMetricsMiddleware(AgentMiddleware):
    """Record orchestrator-level metrics into a :class:`SessionMetrics`.

    Times the overall run, advances the research-cycle index at each
    ``after_agent`` boundary (once per ``RubricMiddleware`` iteration), counts
    the orchestrator's own tool calls, and — by wrapping the ``task`` delegation
    tool — times and counts each sub-agent invocation (the ``task`` call runs the
    whole sub-agent synchronously, so its wall-clock span is the sub-agent's
    execution time). The ``task`` tool is not counted as an ordinary
    orchestrator tool. Purely observational: it never modifies state, messages,
    or tool results, and never returns a ``jump_to`` that would interfere with
    the rubric loop.

    Args:
        metrics: The shared session collector to record into.
    """

    def __init__(self, metrics: SessionMetrics) -> None:
        super().__init__()
        self._metrics = metrics

    def before_agent(self, state: dict[str, Any], runtime: Runtime[Any]) -> None:  # noqa: ARG002
        """Mark the start of the invocation for overall timing."""
        self._metrics._on_run_start()

    async def abefore_agent(self, state: dict[str, Any], runtime: Runtime[Any]) -> None:  # noqa: ARG002
        """Async variant of :meth:`before_agent`."""
        self._metrics._on_run_start()

    def after_agent(self, state: dict[str, Any], runtime: Runtime[Any]) -> None:  # noqa: ARG002
        """Close the current research cycle and advance the cycle index."""
        self._metrics._on_cycle_end()

    async def aafter_agent(self, state: dict[str, Any], runtime: Runtime[Any]) -> None:  # noqa: ARG002
        """Async variant of :meth:`after_agent`."""
        self._metrics._on_cycle_end()

    def _record(self, request: Any, start: float) -> None:
        """Attribute a completed tool call to the orchestrator or a sub-agent."""
        tool_call = request.tool_call
        if tool_call["name"] == _TASK_TOOL_NAME:
            subagent = tool_call.get("args", {}).get(_SUBAGENT_TYPE_ARG)
            if subagent is not None:
                self._metrics._record_subagent_invocation(
                    subagent, time.monotonic() - start
                )
        else:
            self._metrics._record_orchestrator_tool(tool_call["name"])

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Time and count the tool call (a ``task`` call = a sub-agent run)."""
        start = time.monotonic()
        result = handler(request)
        self._record(request, start)
        return result

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> ToolMessage | Command[Any]:
        """Async variant of :meth:`wrap_tool_call`."""
        start = time.monotonic()
        result = await handler(request)
        self._record(request, start)
        return result


class _SubagentMetricsMiddleware(AgentMiddleware):
    """Count one sub-agent's tool calls into a shared :class:`SessionMetrics`.

    One instance is attached per built-in sub-agent. It counts the tool calls
    the sub-agent makes, attributing them to the research cycle in progress on
    the orchestrator. Invocation counts and timings are recorded by
    :class:`_OrchestratorMetricsMiddleware` around the ``task`` call instead (see
    the module docstring for why per-invocation timing cannot live here).

    Args:
        metrics: The shared session collector to record into.
        subagent_name: Name of the sub-agent this middleware is attached to.
    """

    def __init__(self, metrics: SessionMetrics, subagent_name: str) -> None:
        super().__init__()
        self._metrics = metrics
        self._subagent_name = subagent_name

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Count the sub-agent tool call, then delegate to the handler."""
        result = handler(request)
        self._metrics._record_subagent_tool(
            self._subagent_name, request.tool_call["name"]
        )
        return result

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> ToolMessage | Command[Any]:
        """Async variant of :meth:`wrap_tool_call`."""
        result = await handler(request)
        self._metrics._record_subagent_tool(
            self._subagent_name, request.tool_call["name"]
        )
        return result
