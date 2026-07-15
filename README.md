# deep-search-agent

A Python library for deep internet searches (deep search), similar to the deep
research features of ChatGPT and Claude. Built on top of LangChain's
[deepagents](https://docs.langchain.com/oss/python/deepagents).

## Architecture

The `create_deep_search_agent` factory returns a deep agent configured with the
**orchestrator + specialized sub-agents + evaluation loop** pattern:

| Component | Implementation |
|---|---|
| Orchestrator | Main agent (`create_deep_agent`): decomposes the query with `write_todos`, delegates, synthesizes with citations |
| `search-agent` | Web search via SearxNG (+ optional additional search tools), reformulates queries, saves results with their source |
| `fetch-agent` | Downloads and extracts content from URLs: clean HTML with `trafilatura`, PDFs read with `pypdf`, User-Agent from real browsers |
| `fact-check-agent` | Verifies claims against multiple independent sources (has both search and fetch) |
| Shared memory | deepagents virtual filesystem: each sub-agent writes `findings/<source-slug>.md` with URL, date, and claims |
| Evaluator/critic | `RubricMiddleware` (beta): an LLM grader evaluates the answer against a rubric and re-runs the orchestrator up to `max_research_cycles` cycles |

Each sub-agent runs with an isolated context: raw page content does not pollute
the orchestrator's memory; only the synthetic reports and the files in
`findings/` bubble up.

## Installation

```bash
uv sync            # from the repository
# or, as a dependency:
uv add deep-search-agent
```

Requires Python ≥ 3.12. The search tool needs a reachable
[SearxNG](https://docs.searxng.org/) instance with the JSON format enabled
(default: `http://localhost:8888`).

## Quickstart

```python
from deep_search_agent import create_deep_search_agent

agent = create_deep_search_agent(
    model="anthropic:claude-sonnet-4-6",
    searxng_base_url="http://localhost:8888",
    max_research_cycles=3,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "State of the art in quantum error correction in 2026?"}]},
    config={"configurable": {"thread_id": "research-1"}},
)
print(result["messages"][-1].content)
```

The default evaluation rubric (`DEEP_SEARCH_RUBRIC`) is injected automatically:
the refinement loop works with no configuration. For an ad-hoc rubric, pass it
in the invoke state (`{"rubric": "- ..."}`) or to the factory
(`rubric="- ..."`).

## Factory parameters

Deep-search-specific parameters:

| Parameter | Default | Description |
|---|---|---|
| `model` | — (required) | Orchestrator model; inherited by sub-agents and the rubric grader |
| `max_research_cycles` | `3` | Maximum refinement cycles of the evaluator loop (and budget cited in the orchestrator prompt) |
| `max_search_results_per_query` | `5` | Maximum results per search query |
| `max_urls_to_scrape_per_cycle` | `3` | Maximum URLs to fetch per research cycle |
| `searxng_base_url` | `http://localhost:8888` | URL of the SearxNG instance |
| `searxng_engines` | `None` | List of SearxNG engines to restrict the search to |
| `request_timeout` | `15.0` | HTTP timeout (s) for search and fetch |
| `max_content_chars_per_page` | `20000` | Truncation of extracted content per page |
| `search_tools` | `None` | Additional search tools for search-agent and fact-check-agent (e.g. Tavily, RAG retrieval) |
| `rubric` | `DEEP_SEARCH_RUBRIC` | Custom evaluation rubric |
| `auto_rubric` | `True` | Auto-inject the rubric into the state on every invoke |
| `subagents_middleware` | `()` | Extra middleware injected into each built-in sub-agent (search-agent, fetch-agent, fact-check-agent) |
| `subagents` | `None` | Extra sub-agents, added to the built-in ones |
| `backend` | `StateBackend()` | Filesystem backend shared by the orchestrator and every sub-agent |

All other keyword arguments (`tools`, `checkpointer`, `store`, `skills`,
`interrupt_on`, ...) are passed through unchanged to `create_deep_agent`.

## Extensions

### Adding a retrieval agent (RAG) over the internal knowledge base

```python
rag_agent = {
    "name": "rag-agent",
    "description": "Retrieval over the internal company knowledge base",
    "system_prompt": "Search the vector store and save results to findings/.",
    "tools": [my_vector_store_tool],
}

agent = create_deep_search_agent(
    model="anthropic:claude-sonnet-4-6",
    subagents=[rag_agent],
)
```

### Adding search engines

```python
from langchain_tavily import TavilySearch

agent = create_deep_search_agent(
    model="anthropic:claude-sonnet-4-6",
    search_tools=[TavilySearch(max_results=5)],
)
```

### Persistent backend for findings

```python
from deepagents.backends import FilesystemBackend

agent = create_deep_search_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=FilesystemBackend(root_dir="./research", virtual_mode=True),
)
```

## Testing

```bash
uv run pytest
```

The unit tests require neither network nor API keys: HTTP and LLM are simulated.

## Package structure

```
src/deep_search_agent/
├── factory.py       # create_deep_search_agent
├── prompts.py       # orchestrator/sub-agent prompts + default rubric
├── middleware.py    # DefaultRubricMiddleware (rubric auto-injection)
├── subagents.py     # search/fetch/fact-check agent definitions
└── tools/
    ├── search.py    # SearxNG tool
    └── fetch.py     # URL fetch tool (trafilatura + pypdf)
```
