"""Deep search agent library built on LangChain deepagents.

Public API:

- :func:`create_deep_search_agent` — factory returning a configured deep
  agent implementing the orchestrator + specialized sub-agents + rubric-graded
  refinement architecture.
- :data:`DEEP_SEARCH_RUBRIC` — the default grading rubric.
- :class:`DefaultRubricMiddleware` — auto-injects a rubric into the state.
- :class:`SearchBudgetResetMiddleware` — resets the per-cycle search budget at
  each research-cycle boundary.
- :class:`SearchBudget` — thread-safe per-cycle search budget shared with the
  search tool.
- :class:`SessionMetrics` — thread-safe collector of per-cycle and global
  observability metrics (tool calls, sub-agent invocations and timings, overall
  execution time), wired in via ``create_deep_search_agent(metrics=...)``.
  :class:`SubagentStats` and :class:`CycleMetrics` are its typed read models.
- :func:`create_searxng_search_tool` / :func:`create_fetch_url_tool` — the
  tool factories used by the built-in sub-agents, reusable standalone.
"""

from deep_search_agent.factory import create_deep_search_agent
from deep_search_agent.metrics import CycleMetrics, SessionMetrics, SubagentStats
from deep_search_agent.middleware import (
    DefaultRubricMiddleware,
    SearchBudgetResetMiddleware,
)
from deep_search_agent.prompts import (
    DEEP_SEARCH_RUBRIC,
    FACT_CHECK_AGENT_PROMPT,
    FETCH_AGENT_PROMPT,
    ORCHESTRATOR_PROMPT_TEMPLATE,
    SEARCH_AGENT_PROMPT_TEMPLATE,
)
from deep_search_agent.tools import (
    SearchBudget,
    create_fetch_url_tool,
    create_searxng_search_tool,
)

__all__ = [
    "DEEP_SEARCH_RUBRIC",
    "FACT_CHECK_AGENT_PROMPT",
    "FETCH_AGENT_PROMPT",
    "ORCHESTRATOR_PROMPT_TEMPLATE",
    "SEARCH_AGENT_PROMPT_TEMPLATE",
    "CycleMetrics",
    "DefaultRubricMiddleware",
    "SearchBudget",
    "SearchBudgetResetMiddleware",
    "SessionMetrics",
    "SubagentStats",
    "create_deep_search_agent",
    "create_fetch_url_tool",
    "create_searxng_search_tool",
]

__version__ = "0.2.0"
