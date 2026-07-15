"""Benchmark orchestration: run the agent on each question, then judge it.

For every selected question the runner (1) invokes the real deep-search agent
under a wall-clock cap, capturing the final answer, latency, and rough tool
usage, then (2) scores that answer with the independent LLM-as-a-judge, and
finally (3) aggregates per-metric and overall means across all questions.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deep_search_agent import create_deep_search_agent
from benchmark.debug_middleware import DebugMiddleware
from benchmark.config import build_inference_model, build_judge_model
from benchmark.judge import JudgeResult, judge_answer
from benchmark.metrics import METRIC_KEYS
from benchmark.questions import QUESTIONS, QUESTIONS_BY_ID
from deepagents.backends import FilesystemBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from benchmark.config import BenchmarkConfig
    from benchmark.questions import BenchmarkQuestion


@dataclass
class RunResult:
    """The outcome of a single agent invocation (before judging)."""

    answer: str
    duration_s: float
    num_messages: int
    num_tool_calls: int
    error: str | None = None


@dataclass
class QuestionResult:
    """Everything the report needs about one benchmark item."""

    question: BenchmarkQuestion
    run: RunResult
    judge: JudgeResult


@dataclass
class BenchmarkReport:
    """Aggregated results across all questions."""

    results: list[QuestionResult] = field(default_factory=list)
    metric_means: dict[str, float] = field(default_factory=dict)
    overall_mean: float = 0.0
    total_duration_s: float = 0.0


def _message_content_to_text(content: object) -> str:
    """Coerce a LangChain message ``content`` (str or block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _count_tool_calls(messages: Sequence[object]) -> int:
    """Best-effort count of tool calls across the message trace."""
    total = 0
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            total += len(tool_calls)
    return total


def _invoke_agent(
    agent: object,
    question: BenchmarkQuestion,
    config: BenchmarkConfig,
) -> RunResult:
    """Invoke the agent once for ``question``, capped at ``config.agent_timeout``."""
    invoke_config = {
        "recursion_limit": config.recursion_limit,
        "configurable": {"thread_id": f"bench-{question.id}"},
    }
    payload = {"messages": [{"role": "user", "content": question.prompt}]}

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.invoke, payload, config=invoke_config)  # type: ignore[attr-defined]
        try:
            result = future.result(timeout=config.agent_timeout)
        except FuturesTimeout:
            return RunResult(
                answer="",
                duration_s=round(time.perf_counter() - start, 2),
                num_messages=0,
                num_tool_calls=0,
                error=f"agent run exceeded {config.agent_timeout}s wall-clock cap",
            )
        except Exception as exc:  # noqa: BLE001 - one failure must not abort all
            return RunResult(
                answer="",
                duration_s=round(time.perf_counter() - start, 2),
                num_messages=0,
                num_tool_calls=0,
                error=f"agent invocation failed: {exc}",
            )

    duration = round(time.perf_counter() - start, 2)
    messages = result.get("messages", []) if isinstance(result, dict) else []
    answer = _message_content_to_text(messages[-1].content) if messages else ""
    return RunResult(
        answer=answer,
        duration_s=duration,
        num_messages=len(messages),
        num_tool_calls=_count_tool_calls(messages),
    )


def _select_questions(config: BenchmarkConfig) -> list[BenchmarkQuestion]:
    """Resolve ``config.question_ids`` (empty => all) to question objects."""
    if not config.question_ids:
        return list(QUESTIONS)
    selected: list[BenchmarkQuestion] = []
    for qid in config.question_ids:
        if qid not in QUESTIONS_BY_ID:
            known = ", ".join(QUESTIONS_BY_ID)
            msg = f"unknown question id {qid!r}; known ids: {known}"
            raise ValueError(msg)
        selected.append(QUESTIONS_BY_ID[qid])
    return selected


def run_benchmark(
    config: BenchmarkConfig,
    *,
    on_event: Callable[[str], None] | None = None,
) -> BenchmarkReport:
    """Run the full benchmark and return the aggregated report.

    Args:
        config: Resolved benchmark configuration.
        on_event: Optional progress callback (receives human-readable lines);
            defaults to no-op. The CLI passes ``print``.

    Returns:
        A :class:`BenchmarkReport` with per-question results and aggregates.

    Raises:
        ValueError: If ``config.question_ids`` names an unknown question.
        RuntimeError: If ``langchain-openai`` is unavailable.
    """
    log = on_event or (lambda _msg: None)
    questions = _select_questions(config)

    log(f"Building inference model: {config.model}")
    inference_model = build_inference_model(config)
    log(f"Building judge model: {config.judge_model}")
    judge_model = build_judge_model(config)

    agent = create_deep_search_agent(
        model=inference_model,
        middleware=[DebugMiddleware()],
        searxng_base_url=config.searxng_base_url,
        max_research_cycles=config.max_research_cycles,
        max_search_results_per_query=config.max_search_results_per_query,
        max_urls_to_scrape_per_cycle=config.max_urls_to_scrape_per_cycle,
    )

    results: list[QuestionResult] = []
    total_duration = 0.0
    for idx, question in enumerate(questions, start=1):
        log(f"[{idx}/{len(questions)}] {question.id} ({question.capability})")
        run = _invoke_agent(agent, question, config)
        total_duration += run.duration_s
        if run.error:
            log(f"    run error: {run.error}")
        else:
            log(
                f"    answered in {run.duration_s}s "
                f"({run.num_tool_calls} tool calls); judging…"
            )
        judge = judge_answer(judge_model, question, run.answer)
        if judge.error:
            log(f"    judge error: {judge.error}")
        else:
            log(f"    overall {judge.overall}/5")
        results.append(QuestionResult(question=question, run=run, judge=judge))

    metric_means = _aggregate_metric_means(results)
    overall_mean = (
        round(sum(r.judge.overall for r in results) / len(results), 3)
        if results
        else 0.0
    )
    return BenchmarkReport(
        results=results,
        metric_means=metric_means,
        overall_mean=overall_mean,
        total_duration_s=round(total_duration, 2),
    )


def _aggregate_metric_means(results: Sequence[QuestionResult]) -> dict[str, float]:
    """Mean of each metric across all questions."""
    means: dict[str, float] = {}
    if not results:
        return {key: 0.0 for key in METRIC_KEYS}
    for key in METRIC_KEYS:
        values = [
            score.score for r in results for score in r.judge.scores if score.key == key
        ]
        means[key] = round(sum(values) / len(values), 3) if values else 0.0
    return means
