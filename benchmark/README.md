# Deep-search agent benchmark

An **opt-in, real end-to-end** performance benchmark for `deep_search_agent`.
It runs the agent on five deliberately complex research questions using a live
LLM (via OpenRouter) and a live SearxNG instance, then grades each answer with
an independent **LLM-as-a-judge** on four 0–5 metrics.

This is a developer tool: it lives outside `src/` and is never shipped with the
library.

## What it measures

Five questions, each chosen to stress a different part of the pipeline:

| Question id             | Capability exercised                                   |
| ----------------------- | ------------------------------------------------------ |
| `ev-supply-chains`      | Query decomposition + parallel multi-topic synthesis   |
| `central-bank-divergence` | Recency / time-sensitive research                    |
| `ai-water-footprint`    | Contradiction handling + fact-checking                 |
| `ipcc-sea-level`        | Primary-document / PDF fetching + numeric extraction   |
| `solid-state-batteries` | Completeness + honest gap declaration                  |

Each answer is scored 0–5 on four metrics (see `metrics.py`):

- **Completeness** — every part answered, or gaps declared.
- **Factual accuracy** — claims correct and internally consistent.
- **Citation quality / traceability** — every claim maps to a specific source.
- **Coherence & contradiction handling** — organized; reports conflicting sources.

The judge is independent of the agent's internal rubric loop: separate model,
sees only the question and the final answer.

## Requirements

1. Install the extra:

   ```bash
   uv sync --extra benchmark          # or: pip install -e ".[benchmark]"
   ```

2. A reachable **SearxNG** instance (JSON API), e.g. `http://localhost:8888`.

3. An **OpenRouter** API key:

   ```bash
   export OPENROUTER_API_KEY=sk-or-...
   ```

   > Costs real tokens. Both the inference model and the judge model are
   > configurable; `--base-url` can be repointed at any OpenAI-compatible
   > gateway (including a local one, with a placeholder key).

## Usage

```bash
python -m benchmark                       # all 5 questions, defaults
python -m benchmark --list                # show questions and exit
python -m benchmark --questions ipcc-sea-level ai-water-footprint
python -m benchmark \
    --model openai/gpt-4o \
    --judge-model anthropic/claude-opus-4.1 \
    --searxng-url http://localhost:8888 \
    --max-research-cycles 3
```

### Configuration

Every flag has an environment-variable fallback:

| Flag              | Env var                  | Default                          |
| ----------------- | ------------------------ | -------------------------------- |
| `--model`         | `DSA_BENCH_MODEL`        | `anthropic/claude-sonnet-4.5`    |
| `--judge-model`   | `DSA_BENCH_JUDGE_MODEL`  | `openai/gpt-4o`                  |
| `--base-url`      | `DSA_BENCH_BASE_URL`     | `https://openrouter.ai/api/v1`   |
| `--searxng-url`   | `DSA_BENCH_SEARXNG_URL`  | `http://localhost:8888`          |
| (api key)         | `OPENROUTER_API_KEY`     | — (required)                     |

The default model slugs are OpenRouter identifiers; update them if OpenRouter
renames a model.

## Output

Two timestamped files are written to `--output-dir` (default
`benchmark/results/`):

- `benchmark-<UTC>.json` — full machine-readable record (config, every answer,
  every metric score + rationale, latencies).
- `benchmark-<UTC>.md` — human summary: a per-question × per-metric table with
  column means, plus per-question detail and collapsible answers.

## Layout

```
benchmark/
  questions.py   # the 5 questions + capability tags
  metrics.py     # the 4 metrics + 0-5 scoring anchors
  config.py      # BenchmarkConfig + OpenRouter model builders
  judge.py       # LLM-as-a-judge (prompt + tolerant JSON parsing)
  runner.py      # run agent per question, judge, aggregate
  report.py      # JSON + Markdown writers
  __main__.py    # CLI
```
