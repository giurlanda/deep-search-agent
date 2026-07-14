"""Factory for the deep search agent.

:func:`create_deep_search_agent` wires together the multi-agent deep-search
architecture on top of ``deepagents``:

- **Orchestrator** = the main deep agent, instructed to decompose the query,
  delegate to sub-agents, and synthesize a cited answer.
- **Specialized sub-agents** = ``search-agent`` (SearxNG + optional extra
  search tools), ``fetch-agent`` (trafilatura + pypdf), and
  ``fact-check-agent``; callers can add more via ``subagents``.
- **Shared scratchpad** = deepagents' filesystem (pluggable ``backend``),
  where every sub-agent writes ``findings/<source-slug>.md`` files with
  provenance.
- **Evaluator/critic loop** = ``RubricMiddleware`` (deepagents beta), which
  grades the final answer against a rubric and re-runs the orchestrator up to
  ``max_research_cycles`` times; the default deep-search rubric is
  auto-injected unless the caller provides one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents import RubricMiddleware, create_deep_agent

from deep_search_agent.middleware import DefaultRubricMiddleware
from deep_search_agent.prompts import DEEP_SEARCH_RUBRIC, ORCHESTRATOR_PROMPT_TEMPLATE
from deep_search_agent.subagents import (
    build_fact_check_subagent,
    build_fetch_subagent,
    build_search_subagent,
)
from deep_search_agent.tools import create_fetch_url_tool, create_searxng_search_tool
from deep_search_agent.tools.search import DEFAULT_SEARXNG_BASE_URL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import SystemMessage
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph


def _validate_positive(name: str, value: int) -> None:
    """Raise ``ValueError`` unless ``value`` is a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        msg = f"{name} must be a positive integer, got {value!r}"
        raise ValueError(msg)


def create_deep_search_agent(
    *,
    model: str | BaseChatModel,
    max_research_cycles: int = 3,
    max_search_results_per_query: int = 5,
    max_urls_to_scrape_per_cycle: int = 3,
    searxng_base_url: str = DEFAULT_SEARXNG_BASE_URL,
    searxng_engines: Sequence[str] | None = None,
    request_timeout: float = 15.0,
    max_content_chars_per_page: int = 20_000,
    search_tools: Sequence[BaseTool] | None = None,
    rubric: str | None = None,
    auto_rubric: bool = True,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: Sequence[Any] | None = None,
    **create_deep_agent_kwargs: Any,
) -> CompiledStateGraph:
    """Create a deep-search agent (orchestrator + specialized sub-agents).

    The returned agent decomposes the user's query into sub-questions,
    delegates them to isolated sub-agents (web search, content fetching,
    fact checking), accumulates sourced findings on the shared filesystem
    backend, and synthesizes a cited answer. An LLM-as-a-judge loop
    (deepagents' beta ``RubricMiddleware``) grades the answer against a
    rubric and triggers additional research cycles until the rubric is
    satisfied or ``max_research_cycles`` is reached.

    Args:
        model: Model for the orchestrator and, unless overridden per
            sub-agent, for the sub-agents too. The rubric grader inherits
            this same model. Required — either a provider string (e.g.
            ``"anthropic:claude-sonnet-4-6"``) or a ``BaseChatModel``.
        max_research_cycles: Maximum refinement cycles of the evaluator
            loop (``RubricMiddleware.max_iterations``). Also quoted in the
            orchestrator's instructions as its iteration budget.
        max_search_results_per_query: Result budget per search query,
            enforced by the SearxNG tool and quoted in the search agent's
            instructions.
        max_urls_to_scrape_per_cycle: URL-fetch budget per research cycle,
            quoted in the orchestrator's instructions.
        searxng_base_url: Root URL of the SearxNG instance used by the
            built-in search tool.
        searxng_engines: Optional SearxNG engine allowlist
            (e.g. ``["duckduckgo", "wikipedia"]``).
        request_timeout: HTTP timeout (seconds) for both the search and the
            fetch tools.
        max_content_chars_per_page: Truncation limit for content extracted
            by the fetch tool.
        search_tools: Extra search tools (e.g. a Tavily tool or an internal
            RAG retrieval tool) made available to ``search-agent`` and
            ``fact-check-agent`` alongside the built-in SearxNG tool.
        rubric: Custom grading rubric (newline-delimited checklist).
            Defaults to :data:`~deep_search_agent.prompts.DEEP_SEARCH_RUBRIC`.
        auto_rubric: When ``True`` (default), the rubric is auto-injected
            into the invocation state so the evaluation loop works out of
            the box. When ``False``, the loop only activates if the caller
            passes a ``rubric`` key in the invocation state.
        system_prompt: Override for the orchestrator system prompt. Defaults
            to the built-in deep-search orchestrator prompt parametrized
            with the cycle/URL budgets.
        middleware: Extra middleware appended after the rubric middleware.
        subagents: Extra sub-agents (e.g. a RAG retrieval agent over an
            internal knowledge base) added alongside the built-in
            ``search-agent``, ``fetch-agent``, and ``fact-check-agent``.
        **create_deep_agent_kwargs: Any remaining ``create_deep_agent``
            parameter (``tools``, ``backend``, ``checkpointer``, ``store``,
            ``skills``, ``interrupt_on``, ...), passed through unchanged.

    Returns:
        The compiled deep agent graph, ready for ``invoke`` / ``astream``.

    Raises:
        ValueError: If ``model`` is missing, a budget parameter is not a
            positive integer, or a reserved sub-agent name is reused.
    """
    if model is None:
        msg = (
            "create_deep_search_agent requires an explicit `model`: the "
            "rubric grader inherits it and deepagents' implicit default "
            "model is deprecated."
        )
        raise ValueError(msg)
    _validate_positive("max_research_cycles", max_research_cycles)
    _validate_positive("max_search_results_per_query", max_search_results_per_query)
    _validate_positive("max_urls_to_scrape_per_cycle", max_urls_to_scrape_per_cycle)

    # --- Tools for the sub-agents -----------------------------------------
    searxng_tool = create_searxng_search_tool(
        base_url=searxng_base_url,
        engines=searxng_engines,
        timeout=request_timeout,
        max_results=max_search_results_per_query,
    )
    all_search_tools: list[BaseTool] = [searxng_tool, *(search_tools or [])]
    fetch_tool = create_fetch_url_tool(
        timeout=request_timeout,
        max_content_chars=max_content_chars_per_page,
    )

    # --- Sub-agents --------------------------------------------------------
    built_in_subagents = [
        build_search_subagent(
            all_search_tools,
            max_search_results_per_query=max_search_results_per_query,
        ),
        build_fetch_subagent(fetch_tool),
        build_fact_check_subagent(all_search_tools, fetch_tool),
    ]
    reserved_names = {agent["name"] for agent in built_in_subagents}
    extra_subagents = list(subagents or [])
    for agent in extra_subagents:
        name = (
            agent["name"] if isinstance(agent, dict) else getattr(agent, "name", None)
        )
        if name in reserved_names:
            msg = f"subagent name {name!r} is reserved by deep_search_agent"
            raise ValueError(msg)

    # --- Evaluator/critic loop ----------------------------------------------
    effective_rubric = rubric if rubric is not None else DEEP_SEARCH_RUBRIC
    agent_middleware: list[AgentMiddleware] = []
    if auto_rubric:
        # Must precede RubricMiddleware so the rubric is in state when the
        # grading loop initializes.
        agent_middleware.append(DefaultRubricMiddleware(effective_rubric))
    agent_middleware.append(
        RubricMiddleware(model=model, max_iterations=max_research_cycles)
    )
    agent_middleware.extend(middleware)

    # --- Orchestrator --------------------------------------------------------
    if system_prompt is None:
        system_prompt = ORCHESTRATOR_PROMPT_TEMPLATE.format(
            max_research_cycles=max_research_cycles,
            max_urls_to_scrape_per_cycle=max_urls_to_scrape_per_cycle,
        )

    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        middleware=agent_middleware,
        subagents=[*built_in_subagents, *extra_subagents],
        **create_deep_agent_kwargs,
    )
