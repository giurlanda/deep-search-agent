"""Tool factories for the deep search agent's sub-agents."""

from deep_search_agent.tools.fetch import create_fetch_url_tool
from deep_search_agent.tools.search import create_searxng_search_tool

__all__ = ["create_fetch_url_tool", "create_searxng_search_tool"]
