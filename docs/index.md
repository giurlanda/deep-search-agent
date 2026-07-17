# deep-search-agent

A Python library for deep internet searches (deep search), similar to the deep
research features of ChatGPT and Claude. Built on top of LangChain's
[deepagents](https://docs.langchain.com/oss/python/deepagents).

The `create_deep_search_agent` factory returns a deep agent configured with the
**orchestrator + specialized sub-agents + evaluation loop** pattern:

| Component | Implementation |
|---|---|
| Orchestrator | Main agent (`create_deep_agent`): decomposes the query with `write_todos`, delegates, synthesizes with citations |
| `perspective-agent` | Explores the topic from 3-6 distinct angles before decomposition, saved to `/research/perspectives.md`; enabled by default, toggle with `enable_perspectives=False` |
| `search-agent` | Web search via SearxNG (+ optional additional search tools), reformulates queries, saves results with their source |
| `fetch-agent` | Downloads and extracts content from URLs: clean HTML with `trafilatura`, PDFs read with `pypdf`, User-Agent from real browsers |
| `fact-check-agent` | Verifies claims against multiple independent sources (has both search and fetch) |
| Shared memory | deepagents virtual filesystem: each sub-agent writes `findings/<source-slug>.md` with URL, date, and claims |
| Evaluator/critic | `RubricMiddleware` (beta): an LLM grader evaluates the answer against a rubric and re-runs the orchestrator up to `max_research_cycles` cycles |

Each sub-agent runs with an isolated context: raw page content does not pollute
the orchestrator's memory; only the synthetic reports and the files in
`findings/` bubble up.

## Where to go next

- [Installation](installation.md) — install the library and its SearxNG requirement.
- [Quickstart](quickstart.md) — build and invoke your first agent.
- [Architecture](guide/architecture.md) — how the orchestrator, sub-agents, and rubric loop fit together.
- [Extending](guide/extending.md) — add RAG sub-agents, extra search engines, and persistent backends.
- [API Reference](reference/api.md) — the full public API.
