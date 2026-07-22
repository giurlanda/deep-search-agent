# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-07-22

### Added

- GitHub Actions `publish` workflow that builds, tests, and uploads the package
  to PyPI on `v*` tag pushes via PyPI Trusted Publishing (OIDC) — no stored API
  token. See the "Publishing" section of the README for the one-time PyPI
  trusted-publisher setup.

### Changed

- Trimmed the source distribution: it now ships only the package, its tests, and
  `README`/`CHANGELOG`/`LICENSE`, excluding dev tooling (`uv.lock`,
  `.python-version`), CI config, the docs site, and the benchmark harness.

## [0.4.0] - 2026-07-17

### Added

- Opt-in headless-rendering fallback for `fetch_url`. When static extraction
  finds no main content — the usual outcome for JavaScript-only pages and bot
  walls, which previously cost the agent otherwise valid sources — the page is
  re-fetched through a headless Chromium and extracted again. Enable it with
  `create_deep_search_agent(enable_js_render_fallback=True)`; `js_render_timeout`
  (default `30.0` s) bounds the wait for a page to settle ([#23]).

  The fallback is off by default and needs the new `js-render` extra plus the
  browser binary (`pip install deep-search-agent[js-render]` and
  `playwright install chromium`). When either is missing, or rendering fails,
  the tool returns an `ERROR: ...` string as usual instead of raising, so the
  agent reroutes to another source. Rendering runs on a dedicated worker
  thread, keeping Playwright's sync API off any live asyncio loop.

## [0.3.2] - 2026-07-17

### Changed

- `fetch_url` now truncates long documents head+tail instead of keeping only
  the opening. Previously the first `max_content_chars` were kept and the rest
  discarded, which threw away exactly the sections that carry conclusions,
  references and the most recent updates. The tool now keeps 60% of the budget
  from the start and 40% from the end, joined by an explicit marker signalling
  the omitted middle section. Both cuts snap to the nearest paragraph or
  sentence boundary in the outer half of their slice, falling back to a hard
  cut when no boundary qualifies, so neither end reads as a fragment ([#25]).

  The truncation marker changed from a trailing `...(content truncated)` to a
  mid-document `...(middle section omitted for length)...`. Code matching on
  the old marker string needs updating.

## [0.3.1] - 2026-07-17

### Fixed

- The rubric grader no longer flags a phantom "incomplete/truncated" gap on
  long final reports. `deepagents`' `RubricMiddleware` truncates every
  transcript message it sends to the grader at 4,000 chars, which cut off the
  tail of the orchestrator's synthesized answer (the "Gaps & limitations"
  section and the numbered bibliography) and made the grader report a
  completeness gap that did not exist. The factory now wires a new
  `DeepSearchRubricMiddleware` — a thin subclass of `RubricMiddleware` that
  passes the final answer to the grader untruncated while keeping earlier
  transcript messages (e.g. large tool outputs) bounded ([#22]).

### Added

- `DeepSearchRubricMiddleware` is exported from the package `__all__`.
- New `on_evaluation` factory parameter that forwards a callback to the
  evaluator middleware; it is invoked with each `RubricEvaluation` after the
  grader scores a research cycle (e.g. to log per-criterion verdicts or stream
  progress). Exceptions it raises are logged and suppressed.

## [0.3.0] - 2026-07-16

### Added

- New `perspective-agent` sub-agent (`build_perspective_subagent` in
  `subagents.py`), modeled on STORM-style perspective-guided question asking:
  given the research topic, it runs 1-2 exploratory searches and writes 3-6
  distinct perspectives (analysis axes, stakeholder viewpoints, dimensions of
  the problem), each with 2-4 targeted questions, to
  `/research/perspectives.md`.
- New `enable_perspectives` factory parameter (default `True`). When enabled,
  the orchestrator prompt gets a new step 0 that delegates to
  `perspective-agent` before decomposition and builds todos as
  perspective → questions pairs instead of a flat sub-question list. Set to
  `False` to skip the extra delegation cycle for simple, single-axis queries.
- New `PERSPECTIVE_AGENT_PROMPT` prompt template, exported from the package
  `__all__` alongside the other sub-agent prompts.
- `"perspective-agent"` is now a reserved sub-agent name (alongside
  `search-agent`, `fetch-agent`, `fact-check-agent`): a caller-supplied
  sub-agent with that name raises `ValueError`, even when
  `enable_perspectives=False` ([#5]).

### Changed

- **Behavior change on the default path**: `enable_perspectives=True` by
  default adds one extra delegation cycle (and its token cost) to every run,
  without changing any function signature. Pass `enable_perspectives=False`
  to restore the previous single-axis decomposition behavior.

## [0.2.4] - 2026-07-16

### Added

- New `max_query_variants` factory parameter (default `3`, validated as a
  positive integer). It is threaded into `SEARCH_AGENT_PROMPT_TEMPLATE` the
  same way `max_search_results_per_query` is, and quoted in the search agent's
  instructions.

### Changed

- `search-agent` now generates several query variants per sub-question on the
  happy path — synonyms, broader/narrower terms, an English reformulation, or a
  different angle — and issues them as parallel `internet_search` calls in the
  same turn, then deduplicates by URL and keeps the best results across all
  variants. Previously it ran a single query and only reformulated after an
  outright failure, so recall depended on the orchestrator's first phrasing
  ([#6]).

## [0.2.3] - 2026-07-16

### Changed

- The orchestrator's final synthesis is now outline-first. The single
  free-form SYNTHESIZE step in `ORCHESTRATOR_PROMPT_TEMPLATE` is split into
  three explicit steps: **outline** (write `report/outline.md` with sections
  derived from the researched perspectives/sub-questions, depth scaled to the
  question's complexity), **section-by-section synthesis** with numbered `[n]`
  citations, and **assembly** into an executive summary, the sections, a
  required "Gaps & limitations" section, and a numbered "Sources"
  bibliography. Refinement cycles update the outline and keep the citations and
  bibliography consistent. `DEEP_SEARCH_RUBRIC` gains criteria grading the
  executive summary, per-section coverage of every planned
  perspective/sub-question, the explicit gaps section, and a numbered
  bibliography consistent with in-text citations. Prompt-only change; no public
  API change ([#3]).

## [0.2.2] - 2026-07-16

### Added

- `internet_search` (the SearxNG-backed tool built by
  `create_searxng_search_tool`) now accepts optional per-call `category` and
  `time_range` arguments, forwarded to the SearxNG JSON API as `categories`
  and `time_range`. `time_range` is validated against `day`/`week`/`month`/
  `year` and an invalid value returns an `ERROR:` string without hitting the
  network. The `search-agent` prompt now instructs the agent to set
  `time_range` for time-sensitive sub-questions and `category="science"` for
  academic/research ones, so different sub-questions can target recent or
  scholarly sources instead of the same undifferentiated web search ([#4]).

### Changed

- `README.md` now shows license, version, and Python-version badges and links
  to the published documentation on GitHub Pages.

## [0.2.1] - 2026-07-16

### Changed

- The orchestrator, `search-agent`, and `fetch-agent` prompts now maintain and
  consult a shared source index at `findings/_sources.md`: one line per URL
  recording its status (`saved` / `failed` / `discarded`) and the associated
  `findings/<source-slug>.md` file. Sub-agents check it before searching or
  fetching and append their outcomes, and the orchestrator uses it to avoid
  re-running queries or re-fetching pages — on refinement cycles it explicitly
  instructs sub-agents to diversify domains relative to what is already indexed
  ([#2]). Prompt-only change; no public API change.

## [0.2.0] - 2026-07-16

### Added

- `SessionMetrics`, a thread-safe collector of observability metrics, and an
  optional `metrics` parameter on `create_deep_search_agent` to wire it in
  ([#15]). When a `SessionMetrics` instance is passed, observation-only
  middleware injected into the orchestrator and each built-in sub-agent record,
  over the whole session:
  - per research cycle: the orchestrator's tool-call counts, each sub-agent's
    invocation count, and each sub-agent's tool-call counts;
  - globally: total tool-call counts (orchestrator + sub-agents), total
    sub-agent invocations, per-sub-agent execution time (average/min/max), and
    the overall execution time.
  - Metrics accumulate for the lifetime of the object (across research cycles
    and successive invocations of a reused agent) until `reset()` is called;
    read them through typed properties (`cycles`, `global_tool_calls`,
    `subagent_stats`, ...) or as a JSON-serializable mapping via `to_dict()`.
  - New public symbols `SessionMetrics`, `SubagentStats`, and `CycleMetrics`.
    The sub-agent delegation tool (`task`) is tracked as a sub-agent invocation
    rather than as an orchestrator tool. The `metrics` parameter defaults to
    `None` (disabled), so behavior is backward compatible.

## [0.1.4] - 2026-07-15

### Added

- `searxng_rate_limit` and `searxng_budget` parameters on
  `create_deep_search_agent` ([#13]) to throttle SearxNG usage when sub-agents
  run searches concurrently through deepagents' thread pool:
  - `searxng_rate_limit` sets the minimum number of seconds between two SearxNG
    requests, enforced by a thread-safe min-interval limiter shared by the
    built-in search tool. If a request would wait longer than `request_timeout`
    for a free slot, the tool returns an `ERROR:` string instead of performing
    it.
  - `searxng_budget` caps the number of SearxNG searches per research cycle;
    once exhausted the tool returns an `ERROR:` string telling the model no
    budget is left. The counter is reset at each research-cycle boundary by the
    new `SearchBudgetResetMiddleware`.
  - New public symbols `SearchBudget` and `SearchBudgetResetMiddleware`; the
    `create_searxng_search_tool` factory gains `min_request_interval` and
    `budget` parameters. Both new factory parameters default to off/unlimited,
    so behavior is backward compatible.

## [0.1.3] - 2026-07-15

### Added

- `subagents_middleware` parameter on `create_deep_search_agent` ([#11]):
  extra middleware (e.g. logging, rate limiting) injected into each built-in
  sub-agent (`search-agent`, `fetch-agent`, `fact-check-agent`) via their
  `SubAgent.middleware` field. Sub-agents passed via `subagents` are
  caller-owned and left untouched.

## [0.1.2] - 2026-07-15

### Added

- Explicit `backend` parameter on `create_deep_search_agent` ([#9]): the
  factory now resolves a single filesystem backend (defaulting to a shared
  `StateBackend` when none is given, otherwise propagating the caller's
  instance) and hands it to `create_deep_agent`, so the orchestrator and every
  sub-agent provably operate on the same virtual filesystem and
  `findings/<source-slug>.md` files flow back to the orchestrator. Behavior is
  backward compatible; `backend` is no longer an undocumented pass-through
  kwarg.

## [0.1.1] - 2026-07-15

### Added

- Opt-in, real end-to-end `benchmark/` suite: runs the agent on five
  deliberately complex research questions via a live LLM (OpenRouter) and a
  live SearxNG instance, then grades each answer with an independent
  LLM-as-a-judge on four 0-5 metrics. Lives outside `src/` and is never
  shipped with the library; adds a `benchmark` extra (`langchain-openai`).
- MkDocs (Material + mkdocstrings) documentation site, published to GitHub
  Pages via the `docs` workflow: home, installation, quickstart, architecture
  and extending guides, and an auto-generated API reference.
- `[project.urls]` metadata (Homepage, Documentation, Repository, Issues).
- `ruff` in the dev dependency group; the codebase is now linted and formatted
  with it.

### Changed

- Gap-driven refinement cycles ([#1]): on re-entry after a failed rubric
  grading, the orchestrator prompt now instructs the agent to map the grading
  feedback to concrete gaps (recorded in `research/gaps.md`), add targeted
  todos for those gaps only, and re-synthesize reusing the findings already
  collected — instead of restarting the decomposition from scratch.

### Fixed

- Package build: `dynamic = ["version"]` and the correct `[tool.hatch.version]`
  path (`src/deep_search_agent/__init__.py`), so `uv build` and `twine check`
  succeed.
- Classifiers aligned with `requires-python >= 3.12` (dropped 3.11, added 3.13).

## [0.1.0] - 2026-07-14

Initial release.

### Added

- `create_deep_search_agent` factory: orchestrator + specialized sub-agents
  (`search-agent`, `fetch-agent`, `fact-check-agent`) + rubric-graded
  refinement loop, built on LangChain `deepagents`.
- `internet_search` tool over the SearxNG JSON API
  (`create_searxng_search_tool`).
- URL fetch tool extracting HTML with `trafilatura` and PDF with `pypdf`
  (`create_fetch_url_tool`).
- `DefaultRubricMiddleware` for automatic rubric injection, and the default
  `DEEP_SEARCH_RUBRIC`.
- Configurable research budgets, custom rubrics, extra search tools, and extra
  sub-agents; all unrecognized keyword arguments pass through to
  `create_deep_agent`.
- Typed package (`py.typed`), unit test suite (mocked HTTP/LLM), and opt-in
  Playwright end-to-end tests.

[#1]: https://github.com/giurlanda/deep-search-agent/issues/1
[#2]: https://github.com/giurlanda/deep-search-agent/issues/2
[#3]: https://github.com/giurlanda/deep-search-agent/issues/3
[#4]: https://github.com/giurlanda/deep-search-agent/issues/4
[#5]: https://github.com/giurlanda/deep-search-agent/issues/5
[#6]: https://github.com/giurlanda/deep-search-agent/issues/6
[#9]: https://github.com/giurlanda/deep-search-agent/issues/9
[#11]: https://github.com/giurlanda/deep-search-agent/issues/11
[#13]: https://github.com/giurlanda/deep-search-agent/issues/13
[#15]: https://github.com/giurlanda/deep-search-agent/issues/15
[#22]: https://github.com/giurlanda/deep-search-agent/issues/22
[#23]: https://github.com/giurlanda/deep-search-agent/issues/23
[#25]: https://github.com/giurlanda/deep-search-agent/issues/25
[Unreleased]: https://github.com/giurlanda/deep-search-agent/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/giurlanda/deep-search-agent/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/giurlanda/deep-search-agent/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/giurlanda/deep-search-agent/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/giurlanda/deep-search-agent/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/giurlanda/deep-search-agent/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/giurlanda/deep-search-agent/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/giurlanda/deep-search-agent/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/giurlanda/deep-search-agent/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/giurlanda/deep-search-agent/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/giurlanda/deep-search-agent/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/giurlanda/deep-search-agent/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/giurlanda/deep-search-agent/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/giurlanda/deep-search-agent/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/giurlanda/deep-search-agent/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/giurlanda/deep-search-agent/releases/tag/v0.1.0
