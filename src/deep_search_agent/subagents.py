"""Builders for the specialized sub-agents of the deep search architecture.

Each builder returns a :class:`deepagents.SubAgent` mapping (a ``TypedDict``)
ready to be passed to ``create_deep_agent(subagents=...)``:

- ``search-agent``: targeted web search with query reformulation.
- ``fetch-agent``: full-content extraction from URLs (HTML and PDF).
- ``fact-check-agent``: cross-verification of claims against multiple
  sources (gets both search and fetch tools).

Sub-agents run with isolated context windows, so raw page content never
pollutes the orchestrator's memory; only their synthesized reports (and the
``findings/`` files they write through the shared filesystem backend) flow
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
    from langchain_core.tools import BaseTool

SEARCH_AGENT_NAME = "search-agent"
FETCH_AGENT_NAME = "fetch-agent"
FACT_CHECK_AGENT_NAME = "fact-check-agent"


def build_search_subagent(
    search_tools: Sequence[BaseTool],
    *,
    max_search_results_per_query: int,
) -> SubAgent:
    """Build the web-search sub-agent definition.

    Args:
        search_tools: Search tools available to the agent (the SearxNG tool
            plus any user-provided extra search tools).
        max_search_results_per_query: Result budget per query, embedded in
            the agent's instructions.

    Returns:
        A ``SubAgent`` mapping for ``create_deep_agent(subagents=...)``.
    """
    return {
        "name": SEARCH_AGENT_NAME,
        "description": (
            "Runs targeted web searches for a specific sub-question, "
            "reformulates queries when results are poor, and saves sourced "
            "findings to findings/<source-slug>.md. Returns a concise summary "
            "plus the URLs worth a full fetch."
        ),
        "system_prompt": SEARCH_AGENT_PROMPT_TEMPLATE.format(
            max_search_results_per_query=max_search_results_per_query
        ),
        "tools": list(search_tools),
    }


def build_fetch_subagent(fetch_tool: BaseTool) -> SubAgent:
    """Build the fetch/reader sub-agent definition.

    Args:
        fetch_tool: Tool that downloads a URL and extracts its main content
            (HTML via trafilatura, PDFs via pypdf).

    Returns:
        A ``SubAgent`` mapping for ``create_deep_agent(subagents=...)``.
    """
    return {
        "name": FETCH_AGENT_NAME,
        "description": (
            "Downloads specific URLs (HTML pages or PDF documents), extracts "
            "and cleans their main content, and saves relevant claims to "
            "findings/<source-slug>.md. Use when a search snippet is not "
            "enough and the full page must be read."
        ),
        "system_prompt": FETCH_AGENT_PROMPT,
        "tools": [fetch_tool],
    }


def build_fact_check_subagent(
    search_tools: Sequence[BaseTool],
    fetch_tool: BaseTool,
) -> SubAgent:
    """Build the fact-checking sub-agent definition.

    Args:
        search_tools: Search tools used to locate independent sources.
        fetch_tool: Fetch tool used to read those sources in full.

    Returns:
        A ``SubAgent`` mapping for ``create_deep_agent(subagents=...)``.
    """
    return {
        "name": FACT_CHECK_AGENT_NAME,
        "description": (
            "Verifies one or more claims against multiple independent "
            "sources and returns a verdict per claim (confirmed / contested "
            "/ unverifiable) with evidence. Use when sources contradict each "
            "other or a critical claim rests on a single source."
        ),
        "system_prompt": FACT_CHECK_AGENT_PROMPT,
        "tools": [*search_tools, fetch_tool],
    }
