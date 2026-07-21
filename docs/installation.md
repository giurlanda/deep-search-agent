# Installation

## Requirements

- **Python ≥ 3.12**
- A reachable [SearxNG](https://docs.searxng.org/) instance with the JSON
  format enabled (default: `http://localhost:8888`). This powers the built-in
  `internet_search` tool.

## Install the library

```bash
uv add deep-search-agent
```

Or, working from a clone of the repository:

```bash
uv sync
```

## JavaScript rendering fallback (optional)

By default the fetch tool reads the static HTML of a page, so JavaScript-only
pages and bot walls yield no content and are dropped as sources. The
`enable_js_render_fallback` parameter makes the tool re-render those pages in a
headless Chromium before giving up. It needs the `js-render` extra and the
Chromium binary, neither of which is installed by default:

```bash
uv add "deep-search-agent[js-render]"
uv run playwright install chromium
```

```python
agent = create_deep_search_agent(
    model="anthropic:claude-sonnet-4-6",
    enable_js_render_fallback=True,
)
```

The fallback only runs for pages that static extraction could not handle, and
only costs a browser launch when it does. If Playwright or its browsers are
missing, `fetch_url` returns an `ERROR: ...` string like any other fetch
failure rather than raising, so the agent simply reroutes to another source.

## Running a local SearxNG

The built-in search tool talks to SearxNG's JSON API. The quickest way to get
an instance is Docker:

```bash
docker run --rm -d -p 8888:8080 \
  -e "BASE_URL=http://localhost:8888/" \
  -e "SEARXNG_SETTINGS__SEARCH__FORMATS=[html, json]" \
  searxng/searxng
```

!!! note "JSON format is required"
    SearxNG disables the JSON output format by default. Make sure `json` is
    listed under `search.formats` in your instance's `settings.yml` (the
    environment variable above does this for the container), otherwise
    `internet_search` will return an `ERROR: ...` string.

Point the agent at your instance with the `searxng_base_url` parameter (see
[Quickstart](quickstart.md)).

## Model provider credentials

`create_deep_search_agent` requires a `model`. When you pass a provider string
such as `"anthropic:claude-sonnet-4-6"`, the corresponding provider SDK and API
key must be available in the environment (e.g. `ANTHROPIC_API_KEY`). See the
[LangChain model docs](https://docs.langchain.com/oss/python/langchain/models)
for the supported providers.
