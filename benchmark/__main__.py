"""CLI for the deep-search agent benchmark.

Usage::

    export OPENROUTER_API_KEY=sk-or-...
    python -m benchmark                       # all 5 questions, defaults
    python -m benchmark --questions ipcc-sea-level ai-water-footprint
    python -m benchmark --model openai/gpt-4o --judge-model anthropic/claude-opus-4.1
    python -m benchmark --searxng-url http://localhost:8888 --list

Requires a reachable SearxNG instance and a valid OpenRouter API key: the
benchmark runs the agent for real (live LLM + live web search).
"""

from __future__ import annotations

import argparse
import sys

from benchmark.config import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MODEL,
    BenchmarkConfig,
)
from benchmark.questions import QUESTIONS
from benchmark.report import write_reports
from benchmark.runner import run_benchmark


def _build_parser() -> argparse.ArgumentParser:
    defaults = BenchmarkConfig.defaults_from_env()
    parser = argparse.ArgumentParser(
        prog="python -m benchmark",
        description="Run the deep-search agent benchmark (real end-to-end).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=defaults.model,
        help=f"Inference model slug driving the agent (default {DEFAULT_MODEL!r}).",
    )
    parser.add_argument(
        "--judge-model",
        default=defaults.judge_model,
        help=f"Model slug for the LLM-as-a-judge (default {DEFAULT_JUDGE_MODEL!r}).",
    )
    parser.add_argument(
        "--base-url",
        default=defaults.base_url,
        help="OpenAI-compatible endpoint for both models.",
    )
    parser.add_argument(
        "--searxng-url",
        default=defaults.searxng_base_url,
        help="Root URL of the SearxNG instance the agent searches.",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        metavar="ID",
        default=None,
        help="Subset of question ids to run (default: all). See --list.",
    )
    parser.add_argument(
        "--max-research-cycles", type=int, default=defaults.max_research_cycles
    )
    parser.add_argument(
        "--max-search-results-per-query",
        type=int,
        default=defaults.max_search_results_per_query,
    )
    parser.add_argument(
        "--max-urls-to-scrape-per-cycle",
        type=int,
        default=defaults.max_urls_to_scrape_per_cycle,
    )
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=defaults.agent_timeout,
        help="Per-question wall-clock cap in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default=defaults.output_dir,
        help="Directory for the JSON and Markdown reports.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the benchmark questions and exit.",
    )
    return parser


def _print_questions() -> None:
    print("Benchmark questions:\n")
    for q in QUESTIONS:
        print(f"  {q.id}")
        print(f"      capability: {q.capability}")
        print(f"      {q.rationale}\n")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        _print_questions()
        return 0

    config = BenchmarkConfig.defaults_from_env()
    config.model = args.model
    config.judge_model = args.judge_model
    config.base_url = args.base_url
    config.searxng_base_url = args.searxng_url
    config.max_research_cycles = args.max_research_cycles
    config.max_search_results_per_query = args.max_search_results_per_query
    config.max_urls_to_scrape_per_cycle = args.max_urls_to_scrape_per_cycle
    config.temperature = args.temperature
    config.agent_timeout = args.agent_timeout
    config.output_dir = args.output_dir
    config.question_ids = tuple(args.questions) if args.questions else ()

    if not config.api_key:
        print(
            "ERROR: no API key found. Set OPENROUTER_API_KEY (or point "
            "--base-url at a keyless local gateway and export a placeholder).",
            file=sys.stderr,
        )
        return 2

    try:
        report = run_benchmark(config, on_event=print)
    except ValueError as exc:  # bad question id
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # missing langchain-openai
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    json_path, md_path = write_reports(report, config)
    print("\nDone.")
    print(f"  Overall mean: {report.overall_mean}/5")
    print(f"  JSON report:     {json_path}")
    print(f"  Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
