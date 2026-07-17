# CLAUDE.md

Libreria Python **deep-search-agent**: agente di ricerca profonda su internet (stile "deep research"
di ChatGPT/Claude), costruito sopra [`deepagents`](https://docs.langchain.com/oss/python/deepagents)
di LangChain. Espone una factory `create_deep_search_agent` che restituisce un `create_deep_agent`
già configurato.

## Comandi

```bash
uv sync                    # installa dipendenze (incl. dev group)
uv run pytest              # unit test — no rete, no API key (HTTP e LLM simulati)
uv run pytest -m e2e       # test browser end-to-end (richiedono i browser Playwright)
uv add <pkg>               # aggiungi una dipendenza runtime
```

Build backend: `hatchling`. Python richiesto: **≥ 3.12**.

## Architettura

Pattern **orchestratore + sub-agenti specializzati + loop evaluator**. Punti d'ingresso reali:

- [factory.py](src/deep_search_agent/factory.py) — `create_deep_search_agent`, cablaggio di tutto.
- [subagents.py](src/deep_search_agent/subagents.py) — builder di `search-agent`, `fetch-agent`,
  `fact-check-agent` (ritornano `SubAgent` TypedDict di deepagents).
- [middleware.py](src/deep_search_agent/middleware.py) — `DefaultRubricMiddleware` (auto-iniezione rubric).
- [prompts.py](src/deep_search_agent/prompts.py) — prompt di orchestratore/sub-agenti + `DEEP_SEARCH_RUBRIC`.
- [tools/search.py](src/deep_search_agent/tools/search.py) — tool `internet_search` via SearxNG (JSON API).
- [tools/fetch.py](src/deep_search_agent/tools/fetch.py) — tool fetch URL: HTML con `trafilatura`, PDF con `pypdf`.

Flusso: l'orchestratore scompone la query con `write_todos`, delega ai sub-agenti (contesto isolato),
i sub-agenti scrivono `/findings/<source-slug>.md` sul filesystem virtuale di deepagents, poi
l'orchestratore sintetizza con citazioni. `RubricMiddleware` (beta di deepagents) fa da evaluator/critic:
valuta la risposta contro la rubric e rilancia l'orchestratore fino a `max_research_cycles`.

## Convenzioni e vincoli da rispettare

- **Ordine dei middleware**: `DefaultRubricMiddleware` DEVE precedere `RubricMiddleware` — la rubric va
  in `state` *prima* che il loop di grading si inizializzi. Non riordinare.
- **I tool non sollevano eccezioni**: search e fetch ritornano una stringa `"ERROR: ..."` in caso di
  fallimento (rete, timeout, non-2xx), così l'agente può riformulare/reroute senza far crashare il flusso.
  Mantieni questo contratto quando modifichi i tool.
- **`model` è obbligatorio** e keyword-only: niente default implicito (il grader della rubric lo eredita).
  La factory è interamente keyword-only (`*`).
- **Nomi sub-agente riservati**: `search-agent`, `fetch-agent`, `fact-check-agent`. Passare un sub-agente
  extra con uno di questi nomi solleva `ValueError`.
- **Pass-through**: ogni kwarg non riconosciuto va a `create_deep_agent` invariato
  (`checkpointer`, `store`, `skills`, `interrupt_on`, `tools`, ...). Non intercettarli senza motivo.
- **Backend condiviso**: `backend` è un parametro esplicito della factory. La factory risolve un
  unico backend (default `StateBackend()`) e lo propaga a `create_deep_agent`, così orchestratore e
  sub-agenti condividono lo stesso filesystem virtuale (i `findings/*.md` tornano all'orchestratore).
- **Budget positivi**: `max_research_cycles`, `max_search_results_per_query`,
  `max_urls_to_scrape_per_cycle` sono validati come interi positivi (`_validate_positive`).
- **API pubblica**: qualsiasi nuovo simbolo esportato va aggiunto a `__all__` in
  [\_\_init\_\_.py](src/deep_search_agent/__init__.py). Stile docstring: Google-style (Args/Returns/Raises).
- **Tipizzazione**: pacchetto typed (`py.typed`), `from __future__ import annotations` ovunque, import
  pesanti sotto `TYPE_CHECKING`.

## Test

- Unit test in [tests/](tests/): HTTP simulato con la fixture `fake_httpx_get` in
  [conftest.py](tests/conftest.py); nessuna rete né API key.
- Gli e2e sono **opt-in** (`-m e2e`, di default esclusi via `addopts`). Motivo: l'API sync di Playwright
  tiene un loop asyncio sul main thread per l'intera sessione, incompatibile con gli unit test che
  chiamano `asyncio.run()` nello stesso processo. Non rimuovere questa separazione.

## Workflow

Per modifiche tracciate (fix/feature/refactor) usa la skill **project-change-workflow** (issue → branch →
implementazione → test/doc → bump versione + CHANGELOG → commit → PR). Per pubblicare usa **release-pylib**.
La versione è dinamica (`hatchling`): vive in [\_\_init\_\_.py](src/deep_search_agent/__init__.py)
(`__version__`, attualmente 0.1.0) ed è letta via `[tool.hatch.version]` in pyproject.toml.

## Knowledge Graph & Navigation

We use [Graphify](https://graphify.net/) to create a persistent, structured map of this codebase. 
Before executing broad searches or analyzing unfamiliar modules, use the `graphify` query to understand project interconnections.

### Graphify Workflow Guidelines

- **Project Mapping:** Run `/graphify .` in Claude Code to build or update the knowledge graph.
- **Pre-Search Context:** Always check `graphify-out/GRAPH_REPORT.md` first to understand the module map before requesting full file reads.
- **Navigation Shortcuts:** Use commands like `/graphify query "explain the auth flow"` to map out interdependencies rather than using traditional text searches.
- **Auto-Updates:** Keep the graph fresh after major structural refactors using `graphify . --update`.
