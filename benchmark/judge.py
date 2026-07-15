"""LLM-as-a-judge: scores a single agent answer on the four 0-5 metrics.

The judge is deliberately independent from the agent's internal rubric loop:
it uses a separately configured model and only ever sees the question and the
final answer (never the agent's scratchpad). It is asked to return strict JSON,
which is parsed tolerantly so that a chatty model does not crash the run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark.metrics import METRICS, SCALE_MAX, SCALE_MIN

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from benchmark.questions import BenchmarkQuestion


@dataclass
class MetricScore:
    """One metric's outcome."""

    key: str
    score: float
    rationale: str


@dataclass
class JudgeResult:
    """The judge's verdict on a single answer.

    Attributes:
        scores: One :class:`MetricScore` per metric, in ``METRICS`` order.
        overall: Mean of the metric scores (0-5).
        raw: The judge's raw text output, kept for debugging.
        error: Populated if judging itself failed (then ``scores`` is empty).
    """

    scores: list[MetricScore]
    overall: float
    raw: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Serialize for the JSON report."""
        return {
            "overall": self.overall,
            "scores": {
                s.key: {"score": s.score, "rationale": s.rationale} for s in self.scores
            },
            "error": self.error,
        }


def _build_judge_prompt(question: BenchmarkQuestion, answer: str) -> str:
    """Render the judge instruction, embedding the metric anchors and schema."""
    metric_blocks = []
    for m in METRICS:
        anchors = "\n".join(
            f"    - {score}: {meaning}" for score, meaning in sorted(m.anchors.items())
        )
        metric_blocks.append(
            f"- `{m.key}` — {m.name}: {m.description}\n"
            f"  Scoring anchors (interpolate for in-between values):\n{anchors}"
        )
    metrics_doc = "\n".join(metric_blocks)

    schema_keys = ",\n".join(
        f'    "{m.key}": {{"score": <integer {SCALE_MIN}-{SCALE_MAX}>, '
        f'"rationale": "<one or two sentences>"}}'
        for m in METRICS
    )

    return f"""\
You are a strict, impartial evaluator of deep-research answers. You will be \
given a research QUESTION and an ANSWER produced by an automated research \
agent. Score the ANSWER on the following four metrics, each on an integer \
scale from {SCALE_MIN} (worst) to {SCALE_MAX} (best):

{metrics_doc}

Rules:
- Judge only the ANSWER as shown; do not reward intentions or apologies.
- An honestly declared gap is better than a confident fabrication.
- A claim without a specific, checkable source should hurt `citation_quality`.
- If the ANSWER is empty or is an error message, score everything 0.
- Be calibrated: reserve 5 for genuinely excellent work; use the full range.

Return ONLY a single JSON object, no prose, no markdown fences, with exactly \
this shape:
{{
{schema_keys}
}}

QUESTION:
{question.prompt}

ANSWER:
{answer}
"""


def _extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of ``text`` and parse it."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            msg = "no JSON object found in judge output"
            raise ValueError(msg)
        candidate = text[start : end + 1]
    return json.loads(candidate)


def _coerce_score(value: object) -> float:
    """Clamp a parsed score into the [SCALE_MIN, SCALE_MAX] range."""
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(SCALE_MIN)
    return max(float(SCALE_MIN), min(float(SCALE_MAX), score))


def judge_answer(
    judge_model: BaseChatModel,
    question: BenchmarkQuestion,
    answer: str,
) -> JudgeResult:
    """Score one answer with the judge model.

    Never raises: any failure (LLM error, malformed JSON) is captured in the
    returned :class:`JudgeResult.error` with an ``overall`` of 0, so one bad
    item cannot abort the whole benchmark.
    """
    prompt = _build_judge_prompt(question, answer)
    try:
        response = judge_model.invoke(prompt)
        raw = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
    except Exception as exc:  # noqa: BLE001 - report, never crash the run
        return JudgeResult(scores=[], overall=0.0, error=f"judge call failed: {exc}")

    try:
        parsed = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return JudgeResult(
            scores=[], overall=0.0, raw=raw, error=f"judge output not JSON: {exc}"
        )

    scores: list[MetricScore] = []
    for m in METRICS:
        entry = parsed.get(m.key, {})
        if isinstance(entry, dict):
            score = _coerce_score(entry.get("score"))
            rationale = str(entry.get("rationale", "")).strip()
        else:  # tolerate a bare number
            score = _coerce_score(entry)
            rationale = ""
        scores.append(MetricScore(key=m.key, score=score, rationale=rationale))

    overall = sum(s.score for s in scores) / len(scores) if scores else 0.0
    return JudgeResult(scores=scores, overall=round(overall, 3), raw=raw)
