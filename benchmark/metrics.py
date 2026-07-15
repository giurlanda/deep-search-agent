"""The judge metrics (0-5 scale) and their scoring anchors.

Keeping the metric definitions in one place lets the judge prompt, the JSON
schema it must return, and the report columns all derive from the same source
of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

SCALE_MIN = 0
SCALE_MAX = 5


@dataclass(frozen=True)
class Metric:
    """A single 0-5 evaluation dimension.

    Attributes:
        key: Machine key used in the judge's JSON and in the report.
        name: Human-readable name.
        description: What the metric measures.
        anchors: Score -> meaning, used to calibrate the judge (the 0/3/5
            anchors are enough to pin down the scale).
    """

    key: str
    name: str
    description: str
    anchors: dict[int, str]


METRICS: tuple[Metric, ...] = (
    Metric(
        key="completeness",
        name="Completeness",
        description=(
            "Whether the answer addresses every part of the question, or "
            "explicitly declares which parts it could not answer and why."
        ),
        anchors={
            0: "Ignores most of the question or is largely empty.",
            3: "Covers the main parts but leaves clear sub-questions unaddressed "
            "without acknowledging them.",
            5: "Addresses every part, and any genuinely unanswerable part is "
            "explicitly and honestly flagged as a gap.",
        },
    ),
    Metric(
        key="factual_accuracy",
        name="Factual accuracy",
        description=(
            "Whether the concrete claims (facts, figures, dates, names) are "
            "correct and internally consistent, judged against your knowledge "
            "and the plausibility/consistency of the cited evidence."
        ),
        anchors={
            0: "Contains clear factual errors or fabricated figures.",
            3: "Mostly correct but with some dubious or unverifiable claims "
            "stated with unwarranted confidence.",
            5: "Claims are accurate, well qualified, and consistent with the "
            "cited evidence; uncertainty is stated where appropriate.",
        },
    ),
    Metric(
        key="citation_quality",
        name="Citation quality / traceability",
        description=(
            "Whether each factual claim is attributed to a specific source "
            "(URL/document), so the reader can trace it back, with no orphan "
            "claims lacking any source."
        ),
        anchors={
            0: "No citations, or citations that do not map to the claims.",
            3: "Some claims are sourced but several key figures are unattributed "
            "or the sources are vague.",
            5: "Essentially every factual claim carries a specific, checkable "
            "source; sources look relevant and appropriately authoritative.",
        },
    ),
    Metric(
        key="coherence_contradiction",
        name="Coherence & contradiction handling",
        description=(
            "Whether the answer is coherent and well organized, and whether, "
            "when sources disagree, it reports the competing positions instead "
            "of silently picking one."
        ),
        anchors={
            0: "Disorganized or self-contradictory; hides or ignores conflicts.",
            3: "Readable and mostly consistent but flattens real disagreements "
            "into a single unqualified position.",
            5: "Clear structure, and genuine source disagreements are surfaced "
            "and attributed to the conflicting sources.",
        },
    ),
)
"""The four benchmark metrics, all on the same 0-5 scale."""


METRIC_KEYS: tuple[str, ...] = tuple(m.key for m in METRICS)
"""Convenience tuple of metric keys in report order."""
