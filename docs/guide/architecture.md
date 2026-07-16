# Architecture

`deep-search-agent` implements an **orchestrator + specialized sub-agents +
evaluator loop** pattern on top of
[deepagents](https://docs.langchain.com/oss/python/deepagents).

## Flow

0. (Optional, enabled by default) The orchestrator delegates to
   **`perspective-agent`**, which explores the topic from 3-6 distinct angles
   (analysis axes, stakeholder viewpoints, dimensions of the problem) and
   writes them to `research/perspectives.md`. Disable with
   `enable_perspectives=False`.
1. The **orchestrator** (the main deep agent) receives the user query and
   decomposes it into sub-questions with `write_todos` — one per
   perspective/question pair when step 0 ran, otherwise a flat list.
2. It **delegates** each sub-question to a specialized sub-agent, which runs in
   an **isolated context**.
3. Sub-agents write sourced findings to the shared virtual filesystem as
   `findings/<source-slug>.md` (URL, date, claims).
4. The orchestrator **synthesizes** a cited answer from those findings,
   outline-first: it drafts `report/outline.md`, writes each section against the
   relevant findings, then assembles an executive summary, the sections, a gaps
   section, and a numbered bibliography.
5. The **evaluator/critic** (`RubricMiddleware`) grades the answer against a
   rubric and, if it falls short, re-runs the orchestrator — up to
   `max_research_cycles` times.

Because sub-agents are isolated, raw page content never pollutes the
orchestrator's context window; only the synthetic reports and the files in
`findings/` bubble up.

## Components

| Component | Where |
|---|---|
| Orchestrator wiring | [`factory.py`](https://github.com/giurlanda/deep-search-agent/blob/main/src/deep_search_agent/factory.py) — `create_deep_search_agent` |
| Sub-agent builders | [`subagents.py`](https://github.com/giurlanda/deep-search-agent/blob/main/src/deep_search_agent/subagents.py) — `perspective-agent`, `search-agent`, `fetch-agent`, `fact-check-agent` |
| Rubric auto-injection | [`middleware.py`](https://github.com/giurlanda/deep-search-agent/blob/main/src/deep_search_agent/middleware.py) — `DefaultRubricMiddleware` |
| Prompts & default rubric | [`prompts.py`](https://github.com/giurlanda/deep-search-agent/blob/main/src/deep_search_agent/prompts.py) |
| Search tool | [`tools/search.py`](https://github.com/giurlanda/deep-search-agent/blob/main/src/deep_search_agent/tools/search.py) — SearxNG JSON API |
| Fetch tool | [`tools/fetch.py`](https://github.com/giurlanda/deep-search-agent/blob/main/src/deep_search_agent/tools/fetch.py) — HTML via `trafilatura`, PDF via `pypdf` |

## The evaluator loop

The refinement loop is driven by two middleware, and their **order matters**:

1. `DefaultRubricMiddleware` — injects the rubric into the run state. It **must
   run first**, so the rubric is present before the grading loop initializes.
2. `RubricMiddleware` (deepagents beta) — an LLM-as-a-judge that grades the
   answer against the rubric and re-runs the orchestrator until the rubric is
   satisfied or `max_research_cycles` is reached.

The rubric grader inherits the orchestrator's `model`. The default rubric
(`DEEP_SEARCH_RUBRIC`) is used unless you pass your own via the `rubric`
parameter or a `rubric` key in the invoke state. Set `auto_rubric=False` to
disable auto-injection (the loop then only activates when the caller supplies a
`rubric` in the invoke state).

Refinement cycles are **gap-driven**: when the grader re-runs the orchestrator,
its prompt instructs it not to restart the decomposition. Instead it maps each
failed rubric criterion to a concrete gap, records the gaps in
`research/gaps.md` on the shared filesystem, adds targeted todos for those gaps
only, and re-synthesizes — reusing the findings already collected in previous
cycles.

## Design contracts

Two invariants worth knowing when you extend the library:

- **Tools never raise.** The search and fetch tools return an `"ERROR: ..."`
  string on failure (network, timeout, non-2xx) instead of raising, so the
  agent can re-route or reformulate without crashing the run.
- **Reserved sub-agent names.** `search-agent`, `fetch-agent`,
  `fact-check-agent`, and `perspective-agent` are reserved. Passing an extra
  sub-agent that reuses one of these names raises `ValueError` — even when
  `enable_perspectives=False`.
