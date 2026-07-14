# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-07-14

### Added

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
[Unreleased]: https://github.com/giurlanda/deep-search-agent/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/giurlanda/deep-search-agent/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/giurlanda/deep-search-agent/releases/tag/v0.1.0
