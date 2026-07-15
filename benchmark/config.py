"""Benchmark configuration and model construction.

The benchmark talks to two independently configurable models through
`OpenRouter <https://openrouter.ai>`_ (an OpenAI-compatible endpoint, reached
with :class:`langchain_openai.ChatOpenAI` + a custom ``base_url``):

- the **inference model**, driving the deep-search agent (orchestrator,
  sub-agents, and the internal rubric grader inherit it), and
- the **judge model**, used only by the external LLM-as-a-judge that scores the
  final answers.

Both default to OpenRouter model slugs but are overridable via CLI flags or
environment variables, and ``base_url`` can be repointed at any
OpenAI-compatible gateway (including a local one).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# --- Defaults (all overridable) -------------------------------------------
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o"
DEFAULT_SEARXNG_URL = "http://localhost:8888"

# --- Environment variable names -------------------------------------------
ENV_API_KEY = "OPENROUTER_API_KEY"
ENV_BASE_URL = "DSA_BENCH_BASE_URL"
ENV_MODEL = "DSA_BENCH_MODEL"
ENV_JUDGE_MODEL = "DSA_BENCH_JUDGE_MODEL"
ENV_SEARXNG_URL = "DSA_BENCH_SEARXNG_URL"


@dataclass
class BenchmarkConfig:
    """Everything the benchmark run needs, resolved from CLI + environment.

    Attributes:
        model: Inference model slug (OpenRouter) driving the agent.
        judge_model: Model slug used by the LLM-as-a-judge.
        base_url: OpenAI-compatible endpoint for both models.
        api_key: API key for ``base_url`` (never logged).
        searxng_base_url: Root URL of the SearxNG instance the agent searches.
        max_research_cycles: Refinement-loop budget passed to the factory.
        max_search_results_per_query: Per-query result budget.
        max_urls_to_scrape_per_cycle: Per-cycle URL-fetch budget.
        temperature: Sampling temperature for the agent model.
        judge_temperature: Sampling temperature for the judge (kept low).
        request_timeout: Per-request LLM timeout in seconds.
        agent_timeout: Wall-clock cap (seconds) for a single agent run.
        recursion_limit: LangGraph recursion limit per agent invocation.
        output_dir: Where JSON/Markdown reports are written.
        question_ids: Subset of question ids to run (empty = all).
    """

    model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    searxng_base_url: str = DEFAULT_SEARXNG_URL
    max_research_cycles: int = 3
    max_search_results_per_query: int = 5
    max_urls_to_scrape_per_cycle: int = 3
    temperature: float = 0.1
    judge_temperature: float = 0.0
    request_timeout: float = 180.0
    agent_timeout: float = 900.0
    recursion_limit: int = 1000
    output_dir: str = "benchmark/results"
    question_ids: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def defaults_from_env(cls) -> BenchmarkConfig:
        """Build a config seeded from environment variables (CLI overrides it)."""
        return cls(
            model=os.getenv(ENV_MODEL, DEFAULT_MODEL),
            judge_model=os.getenv(ENV_JUDGE_MODEL, DEFAULT_JUDGE_MODEL),
            base_url=os.getenv(ENV_BASE_URL, DEFAULT_BASE_URL),
            api_key=os.getenv(ENV_API_KEY, ""),
            searxng_base_url=os.getenv(ENV_SEARXNG_URL, DEFAULT_SEARXNG_URL),
        )

    def redacted(self) -> dict[str, object]:
        """Config as a dict with the API key removed, safe to write to reports."""
        data = {
            "model": self.model,
            "judge_model": self.judge_model,
            "base_url": self.base_url,
            "searxng_base_url": self.searxng_base_url,
            "max_research_cycles": self.max_research_cycles,
            "max_search_results_per_query": self.max_search_results_per_query,
            "max_urls_to_scrape_per_cycle": self.max_urls_to_scrape_per_cycle,
            "temperature": self.temperature,
            "judge_temperature": self.judge_temperature,
            "request_timeout": self.request_timeout,
            "agent_timeout": self.agent_timeout,
        }
        return data


def _build_chat_model(
    model: str,
    *,
    config: BenchmarkConfig,
    temperature: float,
) -> BaseChatModel:
    """Construct a ``ChatOpenAI`` pointed at the configured OpenAI-compatible API.

    Args:
        model: The provider model slug (e.g. ``"anthropic/claude-sonnet-4.5"``).
        config: The active benchmark config (for ``base_url``/``api_key``).
        temperature: Sampling temperature for this model.

    Returns:
        A ready-to-use chat model.

    Raises:
        RuntimeError: If ``langchain-openai`` is not installed.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - import guard
        msg = (
            "The benchmark requires `langchain-openai`. Install the benchmark "
            "extra: `uv sync --extra benchmark` (or `pip install langchain-openai`)."
        )
        raise RuntimeError(msg) from exc

    return ChatOpenAI(
        model=model,
        base_url=config.base_url,
        api_key=config.api_key or "missing",
        temperature=temperature,
        timeout=config.request_timeout,
        max_retries=2,
    )


def build_inference_model(config: BenchmarkConfig) -> BaseChatModel:
    """Build the model that drives the deep-search agent."""
    return _build_chat_model(
        config.model, config=config, temperature=config.temperature
    )


def build_judge_model(config: BenchmarkConfig) -> BaseChatModel:
    """Build the model used by the LLM-as-a-judge."""
    return _build_chat_model(
        config.judge_model, config=config, temperature=config.judge_temperature
    )
