# quant-agent

A cross-sectional long/short research pipeline on S&P 500 daily price/volume, with an LLM research agent designed to propose and test features on top.

> **👉 See [RESULTS.md](RESULTS.md) for the research log — best configuration, full grid search, comparison tables, and what we learned.**

The pipeline fetches Yahoo Finance daily data, computes a small basket of price/volume features, and runs a vectorized cross-sectional long/short backtest with point-in-time membership reconstruction and sector/size neutralization. The LLM agent layer (Claude Opus 4.7 via `client.beta.messages.tool_runner`) can propose new features, sandbox-exec them, backtest against the current best, and persist findings across runs via a local journal.

## Bias controls

- **Point-in-time S&P 500 membership**, reconstructed by unwinding Wikipedia's "Selected changes" table from today. Only tickers that were constituents on date *d* are eligible to appear in deciles on *d*. Coverage is strong back to ~1990s; earlier dates degrade to today's survivors. Enable via `membership.point_in_time: true` in the config (on by default).
- **Sector + size neutralization** via a single per-date cross-sectional OLS: residualize the signal on `[intercept, sector dummies, log(20d dollar volume)]`. Dollar volume is a size *proxy* — yfinance doesn't give clean point-in-time market cap. Toggle via `neutralize.sector` / `neutralize.size`.

## Install

```bash
cd quant_agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Cache the universe snapshot
quant-agent fetch-universe

# Fetch OHLCV into data/raw/ (parquet, one file per ticker, incremental)
quant-agent fetch-data --start 2014-01-01 --end today --limit 20   # sanity check first
quant-agent fetch-data --start 2014-01-01 --end today              # full run

# Run the sample cross-sectional L/S backtest
quant-agent run-sample --config configs/default.yaml
```

Outputs land under `outputs/sample_run_<ts>/`:

- `summary.json` — Sharpe, vol, drawdown, IC, turnover
- `returns.parquet` — net, gross, turnover per day
- `weights.parquet` — date × ticker weights
- `per_decile_returns.parquet` — for decile-spread charts
- `ic.parquet` — daily Spearman IC
- `equity.png` — net equity curve

## Library layout

- `universe.py` — S&P 500 tickers from Wikipedia + point-in-time membership matrix (via changes table)
- `data.py` — yfinance OHLCV fetch with parquet cache + incremental refresh
- `features.py` — price/volume feature registry (`FEATURES` dict)
- `signals.py` — winsorize, robust z-score, EWMA smoothing, weighted combine, liquidity filter
- `neutralize.py` — cross-sectional sector + size residualization
- `backtest.py` — vectorized decile / sticky-decile / signal-weighted L/S, dollar-neutral, turnover-based costs
- `metrics.py` — Sharpe, drawdown, hit rate, IC, decile spread
- `sandbox.py` — AST-scanned `exec` for LLM-proposed feature functions
- `agent_tools.py` — `ResearchSession` + `@beta_tool` factory for the research agent
- `agent.py` — Claude-driven research loop (tool runner + prompt caching)
- `cli.py` — click entry points

## Research agent (LLM)

Wire Claude Opus 4.7 to the pipeline: it proposes new price/volume features and backtests them against the current best baseline.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
quant-agent research \
  --goal "Propose 2-3 new price/volume features and test if any improve net Sharpe beyond the baseline."
```

How it works:

- A stable `SYSTEM_PROMPT` describes the pipeline, conventions, baseline (net Sharpe ~+0.13), and feature-function contract. Cached via `cache_control: ephemeral`, so repeated runs are mostly served from cache after the first.
- Three tools: `list_features`, `propose_feature(name, python_code, description)`, `run_backtest_tool(feature_weights, halflife_days, weighting, exit_n_deciles)`.
- LLM-generated feature code is AST-scanned (no imports, no I/O, no dunders, no `eval`/`exec`/`getattr`) and `exec`'d in a restricted namespace with only `np`/`pd` available. Runtime shape is validated before registration.
- The `anthropic` SDK's `tool_runner` handles the agentic loop; we iterate and stream text + tool calls to stdout while building a transcript.

Outputs land under `outputs/research_<ts>/`:

- `transcript.json` — full message-level record (content blocks, usage, cache hits)
- `history.json` — every backtest the agent ran, with config + summary metrics
- `feature_<name>.py` — any features the agent proposed (human-readable source)
- `final_report.md` — the agent's closing text

**Cost note.** Opus 4.7 with adaptive thinking is not cheap — a research run with ~10 tool calls and a few features proposed typically costs $0.50–$3 depending on how much the agent thinks. Use `--limit 100` for fast/cheap iteration during development.

### Cross-session memory (the journal)

By default the agent persists its work to `data/research/` so successive runs build on prior findings instead of starting fresh:

- `data/research/features/<name>.py` + `<name>.json` — every feature the agent has proposed. On the next run, each is re-loaded through the same sandbox that gated it originally, so a hand-edited file still cannot smuggle in imports or I/O.
- `data/research/runs.jsonl` — append-only record of every backtest (config + summary + timestamp).
- `data/research/best.json` — current best-ever net-Sharpe run. Updated when a new run exceeds it.

At the start of every research run, a **state recap** is prepended to the user message (not the system prompt, to keep caching intact): current best config + metrics, list of previously-proposed features with their best Sharpe, and a top-N runs table. The agent sees what's already been tried, which cuts down on re-proposing failed ideas.

```bash
# Normal run — picks up prior features + recaps state for the agent:
quant-agent research --goal "..."

# Fresh run — ignores the journal entirely (useful for ablations):
quant-agent research --goal "..." --fresh

# Inspect what's accumulated:
quant-agent journal list

# Wipe it (with confirmation):
quant-agent journal clear
```

Design choices worth knowing:
- **Same-name re-proposal overwrites** with a bumped `revision_count`. The agent can iterate on a feature.
- **"Better"** means strict net-Sharpe improvement — no confidence-interval gating. Noise-prone, but it's what the agent sees and reasons about.
- **Caching invariant**: the system prompt + tool schemas stay byte-stable across runs. The variable state sits in the user turn where it costs nothing on the prefix cache.

## Tests

```bash
pytest
```

## Extending (LLM-readiness)

New features slot into `FEATURES` in [features.py](src/quant_agent/features.py); `signals.combine` and `backtest.run_backtest` are agnostic. An LLM agent layer can propose new feature functions, register them, and re-run the backtest without touching the engine.
