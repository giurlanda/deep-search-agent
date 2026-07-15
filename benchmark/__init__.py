"""Performance benchmark for :mod:`deep_search_agent`.

The benchmark runs the deep-search agent **end-to-end for real** (live LLM via
OpenRouter + a real SearxNG instance) on a fixed set of five deliberately
complex research questions, then grades each answer with an independent
LLM-as-a-judge on four 0-5 metrics.

It is an opt-in developer tool, not part of the shipped library: it lives
outside ``src/`` and is invoked as ``python -m benchmark``.

Entry points:

- :data:`~benchmark.questions.QUESTIONS` — the five benchmark questions, each
  tagged with the agent capability it is meant to exercise.
- :data:`~benchmark.metrics.METRICS` — the four judge metrics (0-5 scale).
- :func:`~benchmark.runner.run_benchmark` — orchestrates run + judging.
- :mod:`benchmark.__main__` — the CLI (``python -m benchmark``).
"""

from __future__ import annotations

from benchmark.config import BenchmarkConfig
from benchmark.metrics import METRICS, Metric
from benchmark.questions import QUESTIONS, BenchmarkQuestion
from benchmark.runner import run_benchmark

__all__ = [
    "METRICS",
    "QUESTIONS",
    "BenchmarkConfig",
    "BenchmarkQuestion",
    "Metric",
    "run_benchmark",
]
