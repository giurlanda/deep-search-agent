"""Middleware that injects a default grading rubric into the agent state.

``deepagents``' :class:`~deepagents.RubricMiddleware` (the evaluator/critic
loop) only activates when the invocation state contains a ``rubric`` key.
:class:`DefaultRubricMiddleware` makes the loop work out of the box: it runs
before ``RubricMiddleware`` and, when the caller did not supply a rubric,
injects the configured default one. A caller-supplied rubric always wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents.middleware.rubric import RubricState
from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


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
