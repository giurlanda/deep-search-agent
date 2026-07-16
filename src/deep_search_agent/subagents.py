"""Builders for the specialized sub-agents of the deep search architecture.

Each builder returns a :class:`deepagents.SubAgent` mapping (a ``TypedDict``)
ready to be passed to ``create_deep_agent(subagents=...)``:

- ``search-agent``: targeted web search that issues several query variants in
  parallel per sub-question, then deduplicates and keeps the best results.
- ``fetch-agent``: full-content extraction from URLs (HTML and PDF).
- ``fact-check-agent``: cross-verification of claims against multiple
  sources (gets both search and fetch tools).

Sub-agents run with isolated context windows, so raw page content never
pollutes the orchestrator's memory; only their synthesized reports (and the
``/findings/`` files they write through the shared filesystem backend) flow
back up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deep_search_agent.prompts import (
    FACT_CHECK_AGENT_PROMPT,
    FETCH_AGENT_PROMPT,
    SEARCH_AGENT_PROMPT_TEMPLATE,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents import SubAgent
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool

SEARCH_AGENT_NAME = "search-agent"
FETCH_AGENT_NAME = "fetch-agent"
FACT_CHECK_AGENT_NAME = "fact-check-agent"


def build_search_subagent(
    search_tools: Sequence[BaseTool],
    *,
    max_query_variants: int,
    max_search_results_per_query: int,
    middleware: Sequence[AgentMiddleware] = (),
) -> SubAgent:
    """Build the web-search sub-agent definition.

    Args:
        search_tools: Search tools available to the agent (the SearxNG tool
            plus any user-provided extra search tools).
        max_query_variants: Number of distinct query reformulations the agent
            issues in parallel per sub-question, embedded in the agent's
            instructions.
        max_search_results_per_query: Result budget per query, embedded in
            the agent's instructions.
        middleware: Extra middleware to attach to this sub-agent (e.g.
            logging, rate limiting), run before deepagents' default
            sub-agent middleware stack.

    Returns:
        A ``SubAgent`` mapping for ``create_deep_agent(subagents=...)``.
    """
    agent: SubAgent = {
        "name": SEARCH_AGENT_NAME,
        "description": (
            "Runs targeted web searches for a specific sub-question, issuing "
            "several query variants in parallel to widen recall, and saves "
            "sourced findings to /findings/<source-slug>.md. Returns a concise "
            "summary plus the URLs worth a full fetch."
        ),
        "system_prompt": SEARCH_AGENT_PROMPT_TEMPLATE.format(
            max_query_variants=max_query_variants,
            max_search_results_per_query=max_search_results_per_query,
        ),
        "tools": list(search_tools),
    }
    if middleware:
        agent["middleware"] = list(middleware)
    return agent


def build_fetch_subagent(
    fetch_tool: BaseTool,
    *,
    middleware: Sequence[AgentMiddleware] = (),
) -> SubAgent:
    """Build the fetch/reader sub-agent definition.

    Args:
        fetch_tool: Tool that downloads a URL and extracts its main content
            (HTML via trafilatura, PDFs via pypdf).
        middleware: Extra middleware to attach to this sub-agent (e.g.
            logging, rate limiting), run before deepagents' default
            sub-agent middleware stack.

    Returns:
        A ``SubAgent`` mapping for ``create_deep_agent(subagents=...)``.
    """
    agent: SubAgent = {
        "name": FETCH_AGENT_NAME,
        "description": (
            "Downloads specific URLs (HTML pages or PDF documents), extracts "
            "and cleans their main content, and saves relevant claims to "
            "/findings/<source-slug>.md. Use when a search snippet is not "
            "enough and the full page must be read."
        ),
        "system_prompt": FETCH_AGENT_PROMPT,
        "tools": [fetch_tool],
    }
    if middleware:
        agent["middleware"] = list(middleware)
    return agent


def build_fact_check_subagent(
    search_tools: Sequence[BaseTool],
    fetch_tool: BaseTool,
    *,
    middleware: Sequence[AgentMiddleware] = (),
) -> SubAgent:
    """Build the fact-checking sub-agent definition.

    Args:
        search_tools: Search tools used to locate independent sources.
        fetch_tool: Fetch tool used to read those sources in full.
        middleware: Extra middleware to attach to this sub-agent (e.g.
            logging, rate limiting), run before deepagents' default
            sub-agent middleware stack.

    Returns:
        A ``SubAgent`` mapping for ``create_deep_agent(subagents=...)``.
    """
    agent: SubAgent = {
        "name": FACT_CHECK_AGENT_NAME,
        "description": (
            "Verifies one or more claims against multiple independent "
            "sources and returns a verdict per claim (confirmed / contested "
            "/ unverifiable) with a one-line rationale, saving the full "
            "evidence to /findings/fact-check-<claim-slug>.md. Use when "
            "sources contradict each other or a critical claim rests on a "
            "single source."
        ),
        "system_prompt": FACT_CHECK_AGENT_PROMPT,
        "tools": [*search_tools, fetch_tool],
    }
    if middleware:
        agent["middleware"] = list(middleware)
    return agent
