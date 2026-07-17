# Quickstart

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
| `max_query_variants` | `3` | Number of parallel query variants the search agent issues per sub-question (synonyms, broader/narrower terms, English variants) to widen recall |
| `max_search_results_per_query` | `5` | Maximum results per search query |
| `max_urls_to_scrape_per_cycle` | `3` | Maximum URLs to fetch per research cycle |
| `searxng_base_url` | `http://localhost:8888` | URL of the SearxNG instance |
| `searxng_engines` | `None` | List of SearxNG engines to restrict the search to |
| `searxng_rate_limit` | `None` | Minimum seconds between SearxNG requests (thread-safe, shared across concurrent searches); `None` disables rate limiting |
| `searxng_budget` | `None` | Maximum SearxNG searches per research cycle; when exhausted the tool returns an `ERROR:` telling the model no budget is left. `None` means unlimited |
| `request_timeout` | `15.0` | HTTP timeout (s) for search and fetch |
| `max_content_chars_per_page` | `20000` | Truncation of extracted content per page |
| `enable_js_render_fallback` | `False` | Re-fetch pages whose static HTML yields no content through a headless Chromium, recovering JavaScript-only pages and bot walls. Requires the `js-render` extra (see [Installation](installation.md#javascript-rendering-fallback-optional)) |
| `js_render_timeout` | `30.0` | Seconds the headless renderer waits for a page to settle; ignored unless the fallback is enabled |
| `search_tools` | `None` | Additional search tools for search-agent and fact-check-agent (e.g. Tavily, RAG retrieval) |
| `rubric` | `DEEP_SEARCH_RUBRIC` | Custom evaluation rubric |
| `auto_rubric` | `True` | Auto-inject the rubric into the state on every invoke |
| `subagents` | `None` | Extra sub-agents, added to the built-in ones |
| `backend` | `StateBackend()` | Filesystem backend shared by the orchestrator and every sub-agent |
| `metrics` | `None` | A `SessionMetrics` collector; when passed, per-cycle and global observability metrics are recorded into it. See [Extending → Collecting session metrics](guide/extending.md#collecting-session-metrics) |

All other keyword arguments (`tools`, `checkpointer`, `store`, `skills`,
`interrupt_on`, ...) are passed through unchanged to `create_deep_agent`. See
the [full API reference](reference/api.md) for details.

!!! tip "Persistence needs a checkpointer"
    The `thread_id` in `config` only carries state across turns when the agent
    was built with a checkpointer/store. See
    [Extending → Persistent backend](guide/extending.md#persistent-backend-for-findings).
