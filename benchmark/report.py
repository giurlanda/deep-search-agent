"""Serialize a :class:`~benchmark.runner.BenchmarkReport` to JSON and Markdown.

The JSON is the machine-readable record (full answers, rationales, config);
the Markdown is a human-facing summary with a per-question x per-metric table,
column means, and per-question detail sections.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark.metrics import METRICS

if TYPE_CHECKING:
    from benchmark.config import BenchmarkConfig
    from benchmark.runner import BenchmarkReport, QuestionResult


def _score_for(result: QuestionResult, key: str) -> float:
    for s in result.judge.scores:
        if s.key == key:
            return s.score
    return 0.0


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def report_to_dict(report: BenchmarkReport, config: BenchmarkConfig) -> dict:
    """Build the JSON-serializable representation of a run."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config.redacted(),
        "aggregates": {
            "overall_mean": report.overall_mean,
            "metric_means": report.metric_means,
            "total_duration_s": report.total_duration_s,
            "num_questions": len(report.results),
        },
        "results": [
            {
                "id": r.question.id,
                "capability": r.question.capability,
                "prompt": r.question.prompt,
                "rationale": r.question.rationale,
                "run": {
                    "duration_s": r.run.duration_s,
                    "num_messages": r.run.num_messages,
                    "num_tool_calls": r.run.num_tool_calls,
                    "error": r.run.error,
                    "answer": r.run.answer,
                },
                "judge": r.judge.as_dict(),
            }
            for r in report.results
        ],
    }


def _markdown(report: BenchmarkReport, config: BenchmarkConfig) -> str:
    lines: list[str] = []
    lines.append("# Deep-search agent benchmark report")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Inference model: `{config.model}`")
    lines.append(f"- Judge model: `{config.judge_model}`")
    lines.append(f"- SearxNG: `{config.searxng_base_url}`")
    lines.append(f"- Questions: {len(report.results)}")
    lines.append(f"- Total agent wall-clock: {report.total_duration_s}s")
    lines.append(f"- **Overall mean score: {report.overall_mean}/5**")
    lines.append("")

    # Summary table: rows = questions, cols = metrics + overall + latency.
    header = ["Question", *[m.name for m in METRICS], "Overall", "Time (s)"]
    lines.append("## Scores")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in report.results:
        cells = [r.question.id]
        cells += [f"{_score_for(r, m.key):.1f}" for m in METRICS]
        cells += [f"{r.judge.overall:.2f}", f"{r.run.duration_s:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    mean_cells = ["**mean**"]
    mean_cells += [f"**{report.metric_means.get(m.key, 0.0):.2f}**" for m in METRICS]
    mean_cells += [f"**{report.overall_mean:.2f}**", "—"]
    lines.append("| " + " | ".join(mean_cells) + " |")
    lines.append("")

    # Per-question detail.
    lines.append("## Per-question detail")
    lines.append("")
    for r in report.results:
        lines.append(f"### {r.question.id} — {r.question.capability}")
        lines.append("")
        lines.append(f"*Why this question:* {r.question.rationale}")
        lines.append("")
        if r.run.error:
            lines.append(f"> **Run error:** {r.run.error}")
            lines.append("")
        if r.judge.error:
            lines.append(f"> **Judge error:** {r.judge.error}")
            lines.append("")
        lines.append(
            f"- Latency: {r.run.duration_s}s · messages: {r.run.num_messages} · "
            f"tool calls: {r.run.num_tool_calls}"
        )
        for m in METRICS:
            score = _score_for(r, m.key)
            rationale = next(
                (s.rationale for s in r.judge.scores if s.key == m.key), ""
            )
            lines.append(f"- **{m.name}: {score:.1f}/5** — {rationale}")
        lines.append("")
        lines.append("<details><summary>Answer</summary>")
        lines.append("")
        lines.append(r.run.answer or "_(empty)_")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def write_reports(
    report: BenchmarkReport,
    config: BenchmarkConfig,
    *,
    output_dir: str | None = None,
) -> tuple[Path, Path]:
    """Write both the JSON and Markdown reports; return their paths.

    Files are timestamped (``benchmark-<UTC>.json`` / ``.md``) so repeated runs
    do not overwrite each other.
    """
    out = Path(output_dir or config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    json_path = out / f"benchmark-{stamp}.json"
    md_path = out / f"benchmark-{stamp}.md"

    json_path.write_text(
        json.dumps(report_to_dict(report, config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report, config), encoding="utf-8")
    return json_path, md_path
