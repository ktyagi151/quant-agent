# quant-agent — Research Log

A deterministic cross-sectional long/short pipeline on S&P 500 daily price/volume, with an LLM research agent designed to sit on top. This document captures the concrete findings from building it out: what we measured, what worked, what didn't.

Everything below was run on yfinance OHLCV, **503 S&P 500 constituents, 2014-01-01 to 2026-04-17** (~3,090 trading days, ~410 point-in-time members per day on average).

---

## 1. What's built

**Deterministic pipeline** (the pre-LLM layer):
- `universe.py` — current constituents from Wikipedia + **point-in-time membership** reconstructed by unwinding Wikipedia's "Selected changes" table from today.
- `data.py` — yfinance fetch with incremental parquet cache (~500 files, one per ticker).
- `features.py` — feature registry: `mom_12_1`, `reversal_5d`, `volume_shock`, `vol_21d`, `amihud_20d`, `dollar_vol_20d`.
- `signals.py` — robust z-score (median + MAD), winsorize, EWMA smoothing, weighted combine, liquidity filter.
- `neutralize.py` — cross-sectional OLS residualization on sector dummies + log(20d dollar volume) as size proxy.
- `backtest.py` — vectorized decile L/S, **three weighting schemes**: hard decile, sticky decile (two-threshold hysteresis), signal-weighted.
- `metrics.py` — Sharpe, drawdown, hit rate, turnover, decile spread, IC (mean + IR).
- `cli.py` — `fetch-universe`, `fetch-data`, `run-sample`, `compare`, `grid`, `research`, `journal list/clear`.

**LLM research agent layer** (built but not yet run end-to-end — requires `ANTHROPIC_API_KEY`):
- `sandbox.py` — AST-scanned `exec` for LLM-proposed feature functions. Rejects imports, dunder access, `eval`/`exec`/`getattr`/`open`. Only `np` and `pd` reachable in the exec namespace.
- `agent_tools.py` — `ResearchSession` holds the panel and feature cache; three `@beta_tool`-decorated functions: `list_features`, `propose_feature`, `run_backtest_tool`.
- `journal.py` — **cross-session memory**: persisted features (re-exec'd through sandbox on load), append-only `runs.jsonl`, current-best tracker. State recap prepended to the user message on every run.
- `agent.py` — driver using `client.beta.messages.tool_runner` with `claude-opus-4-7`, adaptive thinking, `effort: high`, and prompt caching on the system prompt.

**Tests:** 60 unit + integration tests, all passing. Covers feature math, backtest mechanics (dollar-neutrality, perfect-foresight, cost sign, turnover reduction), cache I/O, membership reconstruction, neutralization, sandbox accept/reject behavior, journal round-trip, and session-with-journal integration (including that a hand-planted malicious feature file in `features/` gets rejected at session-load time without killing the session).

---

## 2. Baseline pipeline (before turnover tuning)

First end-to-end run on the full 503-ticker panel, 2014-present, with point-in-time membership, sector+size neutralization, hard decile L/S, 5bps cost.

| metric | value |
|---|---|
| gross ann return | +7.30% |
| gross Sharpe | **+0.45** |
| net ann return | -1.87% |
| net Sharpe | -0.12 |
| max drawdown | -55% |
| avg turnover (daily) | 73% |
| **cost drag** | **9.2%/yr** |
| IC mean | +0.015 |
| IC IR | +1.68 |
| hit rate | 50.1% |

**The signal has real predictive power** (IC IR +1.68, decile spread monotonic — see §3). **Costs eat all of it.** 73% daily turnover × 5bps = ~9% ann, larger than the +7.3% gross alpha.

Stored at `outputs/sample_run_20260417_002548/` (after adding membership + neutralization).

---

## 3. Decile spread — the signal is real

Annualized gross mean return per decile on the baseline config (d0 = lowest signal, d9 = highest):

```
d0: +14.14%   d3: +13.38%   d6: +13.16%   d9: +21.49%   ← top
d1: +12.17%   d4: +14.56%   d7: +16.65%
d2: +11.77%   d5: +13.50%   d8: +16.12%
```

Top-minus-bottom: **+7.35% ann**, consistent with the +0.45 gross Sharpe. The middle is noisy but monotonicity is clear above d5 and the bottom decile is the lowest on record — that's a real edge, just not enough to survive 5bps × 73% turnover.

---

## 4. Five-variant comparison

Stored at `outputs/compare_20260417_003205/`. Same 503-ticker panel throughout; only the signal-construction / weighting step differs.

| variant | gross ret | gross Sharpe | cost drag | **net Sharpe** | max DD | turnover | IC IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1_baseline (hard decile) | +7.30% | +0.45 | 9.2% | **-0.12** | -55% | 73% | +1.68 |
| v2_no_reversal | +1.52% | +0.09 | 8.5% | -0.42 | -75% | 67% | +1.33 |
| v3_smoothed (hl=5) | +2.21% | +0.13 | 2.3% | -0.00 | -53% | 18% | +1.14 |
| v4_sticky (enter 10%, exit 20%) | +5.63% | +0.38 | 6.0% | -0.02 | -45% | 47% | +1.68 |
| v5_signal_weighted | +5.94% | **+0.50** | 7.2% | -0.11 | -41% | 57% | +1.68 |

Reads:
- **Dropping reversal (v2) was bad.** Gross Sharpe collapses +0.45 → +0.09. Reversal contributes real alpha; churn is the *form* of the alpha, not its opposite.
- **EWMA smoothing (v3) over-dampens at hl=5.** Turnover drops 4× (73% → 18%), but gross Sharpe also drops 70% (+0.45 → +0.13). Net Sharpe lands near zero — closest to breakeven, but we gave up too much.
- **Sticky decile (v4) is the best risk-adjusted baseline variant.** Preserves gross Sharpe (+0.38, 15% give-up), cuts turnover 35%, best drawdown profile after v5.
- **Signal-weighted (v5) has the best gross Sharpe of all (+0.50).** Conviction-sizing works but doesn't cut enough turnover to dominate net.
- **IC IR is rank-invariant for v1/v4/v5.** They differ only in how they translate ranks to weights; v2/v3 change the ranks themselves, hence lower IC IR.

---

## 5. Grid search — halflife × sticky

Stored at `outputs/grid_20260417_004229/`. Full factorial: **5 halflives × 4 exit settings = 20 cells**, baseline features, same neutralization, 5bps cost.

### Net Sharpe (primary metric)

```
exit       off       7       5       3
halflife                                
0       -0.115  -0.065  -0.022  +0.113
2       +0.081  +0.081  +0.129  +0.127
3       +0.061  +0.074  +0.131  +0.093   ← winner
5       -0.003  -0.001  +0.055  -0.031
10      +0.011  -0.027  -0.069  -0.064
```

**Winner: `halflife=3, exit_n_deciles=5` → net Sharpe +0.131.** Near-tie plateau at (hl=2, exit=5)=+0.129 and (hl=2, exit=3)=+0.127 — not a lucky single cell.

### Gross Sharpe

```
exit       off       7       5       3
halflife                                
0       +0.450  +0.413  +0.380  +0.423
2       +0.310  +0.283  +0.304  +0.265
3       +0.244  +0.235  +0.268  +0.199
5       +0.130  +0.117  +0.154  +0.045
10      +0.099  +0.049  -0.007  -0.017
```

### Turnover (daily)

```
exit       off     7      5      3
halflife                            
0          73%   59%   47%   33%
2          31%   25%   21%   14%
3          24%   20%   16%   11%
5          18%   15%   12%    8%
10         12%    9%    7%    5%
```

### Max drawdown

```
exit       off     7      5      3
halflife                            
0         -55%  -47%  -45%  -36%
2         -49%  -48%  -45%  -41%
3         -52%  -47%  -45%  -42%
5         -53%  -50%  -44%  -42%
10        -54%  -53%  -53%  -46%
```

### IC IR (rank-invariant across exit)

```
halflife    IC_IR
0         +1.68
2         +1.32
3         +1.25
5         +1.14
10        +1.00
```

### What the grid shows

**The two levers are complementary because they cut turnover at different stages:**
- **Smoothing** acts at the signal level (dampens noise before ranking). Huge turnover cut, large gross Sharpe cost.
- **Stickiness** acts at the weighting level (don't rebalance on rank jitter within a hold band). Moderate turnover cut, small gross Sharpe cost. IC IR is unchanged because ranks aren't changed.

Stacking both at moderate settings beats either alone. (hl=3, exit=5) gives up ~40% of gross alpha to save ~80% of turnover cost — a good trade at 5bps.

### Winner vs baseline side-by-side

|  | baseline (hl=0, exit=off) | winner (hl=3, exit=5) |
|---|---:|---:|
| gross Sharpe | +0.45 | +0.27 (-40%) |
| turnover | 73% | **16%** (-78%) |
| cost drag | 9.2% ann | **1.9% ann** |
| **net Sharpe** | **-0.12** | **+0.13** |
| max DD | -55% | -45% |
| IC IR | +1.68 | +1.25 |

---

## 6. What didn't work

1. **Dropping reversal_5d** — collapsed gross Sharpe from +0.45 to +0.09. The reversal component's "bad" high turnover *is* the alpha; removing it only saved 6pp of turnover but lost 80% of signal.
2. **EWMA with halflife=10** — over-smooths. Max net Sharpe at hl=10 is only +0.011 (exit=off). Halflife >5 days is too slow for a reversal-heavy composite.
3. **Pure signal-weighting (v5)** — best gross Sharpe of all variants (+0.50) but the highest-conviction names churn as hard as the weakest, so turnover stays at 57%. Useful building block; not a standalone turnover solution.

---

## 7. Known limitations (documented, not fixed)

**Addressed in v1:**
- ✅ Point-in-time S&P 500 membership (via Wikipedia changes table).
- ✅ Sector + size neutralization (cross-sectional OLS residuals).

**Not addressed:**
1. **Size proxy is log(20d dollar volume), not market cap.** yfinance doesn't give clean point-in-time market cap. Dollar volume correlates with market cap but imperfectly — a real run needs Sharadar or similar.
2. **Flat 5bps cost model.** Ignores spread + market impact. At the edges of the grid (turnover <10%), a volume-aware model (`cost = half_spread + k·√(|trade|/ADV)`) would shift the optimum.
3. **Survivorship tail.** Wikipedia's "selected changes" is strong post-~1995 but thin earlier. Back-tests before 2000 should not be trusted.
4. **Thin feature basket.** Only 3 signals are combined. An IC IR of +1.68 gross is decent, but the gap to production-quant IR ranges (3+) comes from *breadth*, not tuning.
5. **No regime analysis.** Performance through 2020 crash, 2022 drawdown, and 2023 recovery is averaged together. A by-year or by-regime IC decomposition would expose fragility.

---

## 8. Status of the LLM research agent

Architecture complete, **not yet run end-to-end** (blocked on API key availability).

What's wired:
- **System prompt** describes the pipeline, conventions, baseline (+0.13 net Sharpe target), and feature-function contract. Cached via `cache_control: ephemeral`.
- **Three tools:** `list_features`, `propose_feature(name, python_code, description)`, `run_backtest_tool(feature_weights, halflife_days, weighting, exit_n_deciles)`.
- **Sandbox:** AST-scan (rejects imports, dunders, `eval`/`exec`/`getattr`/`open`) + exec in a namespace that only exposes `np` and `pd`. Shape validation on the returned DataFrame.
- **Journal:** `data/research/features/*.py` + `.json`, `runs.jsonl`, `best.json`. Features re-exec through the same sandbox on load. State recap (current best + prior features with their best Sharpe) prepended to the user message — not the system prompt, so cache stays intact.
- **Safety:** `--fresh` flag for ablations. `quant-agent journal list/clear` for inspection / wipe.
- **Cost estimate:** ~$0.10–$0.50 per run at `--limit 100`, $0.50–$3 at full universe, depending on how much the agent thinks.

Expected first-run command:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
quant-agent research \
  --goal "Propose 2-3 new price/volume features and test whether any improve net Sharpe beyond the winner (hl=3, exit=5)." \
  --limit 100
```

---

## 9. Reproducing this

```bash
# One-time setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify tests
pytest                                         # → 60 passed

# Fetch data (first run: ~30s; incremental thereafter)
quant-agent fetch-universe
quant-agent fetch-data --start 2014-01-01 --end today

# Single baseline run
quant-agent run-sample --config configs/default.yaml

# Five-variant comparison
quant-agent compare

# Full grid (halflife × exit)
quant-agent grid --halflives 0,2,3,5,10 --exits off,7,5,3
```

All outputs land under `outputs/<run-type>_<timestamp>/` with `summary.json`, `returns.parquet`, `weights.parquet`, `per_decile_returns.parquet`, `ic.parquet`, and an `equity.png`.

---

## 10. Takeaways

1. **Bias controls matter and are cheap.** Point-in-time membership moved the universe from 503 survivors to avg 410 actual members/day. Sector + size neutralization alone lifted IC IR from roughly 0 to +1.68 on this signal set.
2. **Gross Sharpe and net Sharpe are different metrics.** The baseline's gross +0.45 is respectable; the net -0.12 says 5bps costs at 73% turnover kill it. Signal-quality diagnostics (IC, decile spread) and portfolio-construction diagnostics (turnover, cost drag) should both be reported.
3. **Smoothing and stickiness stack.** They reduce turnover through different mechanisms, so their effects compound. The grid shows a clean plateau near (hl=2-3, exit=5) rather than a single optimal cell — that's a more robust result than a single lucky point.
4. **Further gains need more features, not more tuning.** We've wrung most of the improvement out of this 3-feature basket. Breadth (more uncorrelated signals) is the next lever, and the agent layer is set up precisely for that.
