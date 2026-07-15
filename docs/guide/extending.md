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

## Collecting session metrics

Pass a `SessionMetrics` instance to observe how much work the orchestrator and
its sub-agents did. The collector is thread-safe and accumulates over every
research cycle (and across successive invocations of the same agent) until you
call `reset()`:

```python
from deep_search_agent import create_deep_search_agent, SessionMetrics

metrics = SessionMetrics()
agent = create_deep_search_agent(
    model="anthropic:claude-sonnet-4-6",
    metrics=metrics,
)

agent.invoke({"messages": [{"role": "user", "content": "..."}]})

# Typed access:
print(metrics.total_duration)          # overall execution time (s)
print(metrics.global_tool_calls)       # {tool_name: count} across orchestrator + sub-agents
print(metrics.global_subagent_invocations)
for name, stats in metrics.subagent_stats.items():
    print(name, stats.count, stats.avg_time, stats.min_time, stats.max_time)
for cycle in metrics.cycles:           # per research cycle
    print(cycle.orchestrator_tool_calls, cycle.subagent_invocations)

# Or a JSON-serializable snapshot:
import json
print(json.dumps(metrics.to_dict(), indent=2))
```

What is recorded, per research cycle and globally:

- how many times the orchestrator called each of its own tools (the `task`
  delegation tool is tracked as a sub-agent invocation instead);
- how many times each sub-agent was invoked;
- how many times each sub-agent called each of its tools;
- per-sub-agent execution time (average, min, max) and the overall execution
  time.

## Reusing the tools standalone

The tool factories are part of the public API and can be used outside the agent:

```python
from deep_search_agent import create_searxng_search_tool, create_fetch_url_tool

search = create_searxng_search_tool(base_url="http://localhost:8888")
fetch = create_fetch_url_tool(max_content_chars=20_000)
```
