"""Middleware for the deep-search agent.

- :class:`DefaultRubricMiddleware` injects a default grading rubric into the
  agent state so ``deepagents``' :class:`~deepagents.RubricMiddleware` (the
  evaluator/critic loop, which only activates when the invocation state
  contains a ``rubric`` key) works out of the box. It runs before
  ``RubricMiddleware`` and, when the caller did not supply a rubric, injects
  the configured default one. A caller-supplied rubric always wins.
- :class:`SearchBudgetResetMiddleware` resets a :class:`SearchBudget` at each
  research-cycle boundary so the per-cycle search budget is replenished every
  time ``RubricMiddleware`` loops the orchestrator back for another cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents.middleware.rubric import RubricState
from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

    from deep_search_agent.tools.search import SearchBudget


class DefaultRubricMiddleware(AgentMiddleware):
    """Inject a default ``rubric`` into the state when none was provided.

    Must be placed *before* ``RubricMiddleware`` in the middleware list so
    the rubric is already in the state when the grading loop initializes.

    Args:
        rubric: The rubric text to inject when the invocation state has no
            ``rubric`` key (or an empty one).
    """

    state_schema = RubricState

    def __init__(self, rubric: str) -> None:
        super().__init__()
        if not rubric or not rubric.strip():
            msg = "DefaultRubricMiddleware requires a non-empty rubric"
            raise ValueError(msg)
        self.rubric = rubric

    def _inject(self, state: RubricState) -> dict[str, Any] | None:
        if state.get("rubric"):
            return None
        return {"rubric": self.rubric}

    def before_agent(
        self,
        state: RubricState,
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Return a state update with the default rubric, or None.

        Args:
            state: Current agent state.
            runtime: Agent runtime (unused).

        Returns:
            ``{"rubric": <default>}`` when the state has no rubric,
            otherwise ``None`` (caller-supplied rubric is preserved).
        """
        return self._inject(state)

    async def abefore_agent(
        self,
        state: RubricState,
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`before_agent`."""
        return self._inject(state)


class SearchBudgetResetMiddleware(AgentMiddleware):
    """Reset a shared :class:`SearchBudget` at every research-cycle boundary.

    The search budget lives inside the ``internet_search`` tool closure (so it
    is shared across the concurrent sub-agents that call it), but the notion
    of a *research cycle* only exists at the orchestrator level: each pass of
    ``RubricMiddleware`` runs the model loop, grades it, and — on
    ``needs_revision`` — jumps back to the model for another cycle.

    This middleware ties the two together by resetting the budget:

    - in :meth:`before_agent`, so the first cycle (and every fresh invocation
      of a reused agent) starts with the full budget; and
    - in :meth:`after_agent`, which fires once per cycle and therefore
      replenishes the budget before ``RubricMiddleware`` may loop back for the
      next one. Sub-agent searches have already completed by then (their tool
      calls block the orchestrator), so resetting here never races an
      in-flight search.

    Args:
        budget: The :class:`SearchBudget` shared with the ``internet_search``
            tool.
    """

    def __init__(self, budget: SearchBudget) -> None:
        super().__init__()
        self.budget = budget

    def before_agent(
        self,
        state: dict[str, Any],  # noqa: ARG002
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> None:
        """Reset the budget at the start of the run (first cycle)."""
        self.budget.reset()

    async def abefore_agent(
        self,
        state: dict[str, Any],  # noqa: ARG002
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> None:
        """Async variant of :meth:`before_agent`."""
        self.budget.reset()

    def after_agent(
        self,
        state: dict[str, Any],  # noqa: ARG002
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> None:
        """Replenish the budget so the next research cycle starts fresh."""
        self.budget.reset()

    async def aafter_agent(
        self,
        state: dict[str, Any],  # noqa: ARG002
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> None:
        """Async variant of :meth:`after_agent`."""
        self.budget.reset()
