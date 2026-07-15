# Extending

Every keyword argument `create_deep_search_agent` does not recognize is passed
through unchanged to `create_deep_agent` (`tools`, `checkpointer`, `store`,
`skills`, `interrupt_on`, ...), so the full deepagents surface stays available.

## Adding a retrieval agent (RAG) over an internal knowledge base

Extra sub-agents are added alongside the built-in ones. They may not reuse the
reserved names (`search-agent`, `fetch-agent`, `fact-check-agent`).

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

## Adding search engines

`search_tools` are made available to both `search-agent` and
`fact-check-agent`, alongside the built-in SearxNG tool.

```python
from langchain_tavily import TavilySearch

agent = create_deep_search_agent(
    model="anthropic:claude-sonnet-4-6",
    search_tools=[TavilySearch(max_results=5)],
)
```

## Persistent backend for findings

The orchestrator and every sub-agent share a single `backend`, so the
`findings/<source-slug>.md` files a sub-agent writes flow back to the
orchestrator on the same filesystem. By default that backend is an ephemeral
`StateBackend`; pass an explicit `backend` to point the agent at a real
directory with a `FilesystemBackend`:

```python
from deepagents.backends import FilesystemBackend

agent = create_deep_search_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=FilesystemBackend(root_dir="./research", virtual_mode=True),
)
```

## Restricting SearxNG engines

```python
agent = create_deep_search_agent(
    model="anthropic:claude-sonnet-4-6",
    searxng_engines=["duckduckgo", "wikipedia"],
)
```

## Customizing the rubric

Pass a newline-delimited checklist as `rubric` to change how the evaluator
grades answers:

```python
agent = create_deep_search_agent(
    model="anthropic:claude-sonnet-4-6",
    rubric=(
        "- Every claim is backed by at least two independent sources\n"
        "- All sources are dated and linked\n"
        "- The answer states remaining uncertainties explicitly"
    ),
)
```

## Reusing the tools standalone

The tool factories are part of the public API and can be used outside the agent:

```python
from deep_search_agent import create_searxng_search_tool, create_fetch_url_tool

search = create_searxng_search_tool(base_url="http://localhost:8888")
fetch = create_fetch_url_tool(max_content_chars=20_000)
```
