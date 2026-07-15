"""The five benchmark questions.

Each question is intentionally complex and picked to stress a *different*
facet of the deep-search architecture (decomposition, recency handling,
contradiction handling, document/PDF fetching, gap declaration). The
``capability`` tag documents which facet the question is meant to exercise so
that a low score can be traced back to a concrete part of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkQuestion:
    """A single benchmark item.

    Attributes:
        id: Short stable slug, used in filenames and CLI selection.
        capability: The agent capability this question primarily exercises.
        prompt: The natural-language query handed to the agent verbatim.
        rationale: Why this question stresses ``capability`` (for the report).
    """

    id: str
    capability: str
    prompt: str
    rationale: str


QUESTIONS: tuple[BenchmarkQuestion, ...] = (
    BenchmarkQuestion(
        id="ev-supply-chains",
        capability="Query decomposition + parallel multi-topic synthesis",
        prompt=(
            "Compare the electric-vehicle battery supply chains of the European "
            "Union, the United States, and China. For each of the three regions, "
            "identify: (a) the main sources of raw and battery-grade refined "
            "lithium and other critical cathode materials, (b) the largest "
            "domestic battery-cell manufacturers, and (c) the key policy measures "
            "(subsidies, tariffs, local-content or sourcing rules) enacted since "
            "2022. Then highlight the two or three points where the regions most "
            "directly compete or create supply-chain dependencies on each other. "
            "Attribute every figure to a specific source."
        ),
        rationale=(
            "A 3x3 matrix of independent sub-questions rewards decomposing into "
            "parallel search tasks and synthesizing a cross-cutting comparison "
            "with per-claim citations."
        ),
    ),
    BenchmarkQuestion(
        id="central-bank-divergence",
        capability="Recency / time-sensitive research",
        prompt=(
            "Summarize the current monetary-policy stance of the US Federal "
            "Reserve and the European Central Bank. Report each institution's "
            "most recent published policy-rate decision (the date, the resulting "
            "target rate or range, and the stated rationale), describe how their "
            "current stances diverge, and note the forward guidance each has "
            "given for the coming months. Make the as-of date of every figure "
            "explicit and flag anything you could only source to older data."
        ),
        rationale=(
            "Answers go stale fast, so this rewards preferring recent sources, "
            "surfacing publication dates, and explicitly dating each claim."
        ),
    ),
    BenchmarkQuestion(
        id="ai-water-footprint",
        capability="Contradiction handling + fact-checking",
        prompt=(
            "Published estimates of the water and energy footprint of large-scale "
            "AI data centers vary wildly. Gather the range of credible published "
            "estimates for (1) the water consumed per typical large-language-model "
            "query and (2) the water and energy consumed by a single frontier "
            "model training run. Explain the main methodological reasons the "
            "figures disagree so much (e.g. on-site vs off-site water, regional "
            "grid mix, direct vs embodied consumption). Report the competing "
            "positions side by side rather than settling on one number."
        ),
        rationale=(
            "Sources genuinely disagree here, so this exercises the "
            "fact-check-agent and the rubric requirement to present both sides "
            "instead of silently picking one."
        ),
    ),
    BenchmarkQuestion(
        id="ipcc-sea-level",
        capability="Primary-document / PDF fetching + numeric extraction",
        prompt=(
            "Using primary documents (official IPCC reports, ideally the AR6 "
            "Working Group I report and its Summary for Policymakers), report the "
            "projected global mean sea-level rise by 2100 relative to a "
            "1995-2014 baseline under a low-emission scenario (SSP1-2.6) and a "
            "high-emission scenario (SSP5-8.5). Give the numeric likely ranges in "
            "metres and the stated likelihood/confidence language for each, and "
            "cite the specific report and section the numbers come from."
        ),
        rationale=(
            "The authoritative numbers live in long official PDFs, so this "
            "rewards delegating to the fetch-agent (PDF parsing) and extracting "
            "exact figures with confidence qualifiers and section-level "
            "attribution."
        ),
    ),
    BenchmarkQuestion(
        id="solid-state-batteries",
        capability="Completeness + honest gap declaration",
        prompt=(
            "Assess the near-term commercial viability of solid-state batteries "
            "for electric vehicles. Address all four of these points: (1) which "
            "companies have announced pilot-line or gigafactory-scale production "
            "and on what timelines; (2) what independently verified energy-density "
            "and cycle-life figures exist, contrasted against manufacturers' own "
            "claims; (3) the main unsolved technical obstacles (e.g. dendrites, "
            "interfacial resistance, manufacturing scale-up); and (4) explicitly "
            "state which of the above you could NOT substantiate from reliable, "
            "independent sources and why."
        ),
        rationale=(
            "The fourth sub-point directly tests whether the agent declares gaps "
            "instead of fabricating, while (2) tests separating verified data "
            "from vendor claims."
        ),
    ),
)
"""The frozen set of benchmark questions, one per exercised capability."""


QUESTIONS_BY_ID: dict[str, BenchmarkQuestion] = {q.id: q for q in QUESTIONS}
"""Lookup used by the CLI ``--questions`` selector."""
