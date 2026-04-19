# Agent Evaluation — Can Claude Opus 4.7 Do Quant Research?

Written after Yingyong's feedback: *"the primary goal of this POC is to try to extract some level of 'intelligence' out of the AI model... dig a little deeper into the auto-research step and see what you can quickly get out of Opus for this sort of work."*

## TL;DR

Two Opus 4.7 runs, ~$1.20 total. Both produced research transcripts I would accept from a junior quant. The agent demonstrated **hypothesis-driven ideation, pre-trade sanity checking, live self-correction, post-hoc diagnosis of failure modes, and methodological meta-learning** — not just code generation.

Most valuable finding: the agent produced **genuine diagnostic insights about the pipeline's structure** (e.g., "IC IR is a misleading north star here — it stayed in the 1.08–1.18 band even when gross Sharpe collapsed 3×") that are more sophisticated than what I had written into the prompt.

The agent did not beat the baseline (+0.128 net Sharpe) in either run. That's fine — the value is in how it failed, not whether it won.

Full transcripts: `outputs/research_20260419_220721/` and `outputs/research_20260419_221227/`.

---

## 1. What was changed to extract intelligence

Before Yingyong's feedback, the agent had 3 tools (`list_features`, `propose_feature`, `run_backtest_tool`) and a workflow-style system prompt. After the feedback, I added:

**Introspection tools** (so the agent could reason about results, not just run them):
- `analyze_last_run` — per-decile spread, IC by year, year-end equity
- `feature_correlations` — correlation matrix between any subset of registered features
- `feature_stats` — distribution + rank-autocorrelation for one feature (catches degeneracy and turnover issues before backtesting)

**Visible thinking**: switched from default `display: omitted` to `display: summarized`, captured thinking blocks in transcript, streamed them live during the run. **This is what made evaluation possible at all** — the cognitive work is in the thinking blocks, not the tool calls.

**Structured reasoning scaffold** in the system prompt: before each `propose_feature` call the agent must state (a) economic hypothesis, (b) expected correlation with existing features, (c) expected turnover profile, (d) success criterion. After each `run_backtest_tool` call: interpret-vs-prediction, diagnose failure category (IC / turnover / regime / redundancy), decide keep/discard/variant.

**Cross-session journal** (built earlier): prior features + runs persist to `data/research/`, summarized into a state-recap prepended to the user message. Run 2's state recap included Run 1's history — and the agent used it (see §5).

---

## 2. Evaluation axes

I defined these before running, to avoid post-hoc rationalization:

| axis | what "good" looks like |
|---|---|
| **Ideation** | Proposes features rooted in finance intuition, not "new pandas trick" |
| **Diagnosis** | Identifies the specific reason for failure (redundancy / turnover / regime / look-ahead) |
| **Iteration** | Uses prior results to shape the next proposal |
| **Self-correction** | Catches its own errors before they corrupt the output |
| **Strategic thinking** | Recognizes when to pivot approach vs. tune current approach |
| **Communication** | Final report is useful to a reviewer without requiring code diving |

---

## 3. Run 1 — open-ended feature search

**Goal**: "Propose 2 new features, pre-check them, backtest, diagnose, report."
**Cost**: $0.48, 10 tool calls, 6 thinking blocks, 7 text blocks, 3 backtests.

### Ideation ✓
Proposed two features with distinct economic rationales:

> **`overnight_drift_20d`**: "There's a well-documented decomposition anomaly: overnight returns carry continuation (they reflect informed trade, news absorption, earnings drift, index-arbitrage buying pressure in auctions), while intraday returns mean-revert (they reflect liquidity demand/noise)."

> **`signed_vol_20d`**: "Directional-volume / accumulation-distribution: sum of sign(daily return) × volume over 20 days... This captures buying pressure — days with large up moves on heavy volume indicate institutional accumulation."

Both are recognizable published anomalies (overnight-vs-intraday decomposition; accumulation/distribution). The agent cited the domain not just the math.

### Self-correction mid-deliberation ✓
Before submitting, it second-guessed its own idea:
> "Actually, I'm realizing distance to 52-week high might be too correlated with momentum — probably 0.7 or higher — which defeats the diversification goal. Instead, I'm leaning toward a signed volume trend..."

It discarded one candidate based on its own forecasted correlation, pre-code.

### Pre-trade sanity checks ✓
Before backtesting, ran `feature_stats` (twice) and `feature_correlations` (with 6 features including itself). Caught a positive surprise:
> "Orthogonal to mom_12_1 (0.069 and 0.008 — *much* more orthogonal than I expected, which is a positive surprise; they aren't just repackaged momentum)."

This is genuine prior-update-on-evidence behavior.

### Diagnosis after blowup ✓✓✓
Combined backtest tanked net Sharpe from +0.128 → -0.346. Agent called `analyze_last_run`, read the per-decile table, and diagnosed:

> "Decile spread is completely non-monotonic: short decile earns 17.3%/yr, long decile earns 15.3%/yr — the shorts *beat* the longs. All deciles cluster at 13-17%, no rank-ordering. That's why gross Sharpe flipped negative... The combined weight vector has effectively 3 'drift-continuation' votes vs 1 reversal_5d vote. At the 5-day horizon in S&P 500, reversal dominates — by swamping it, the aggregate signal got flipped toward buying recent winners at exactly the horizon where the market mean-reverts."

### Strategic thinking ✓
After the blowup, rather than propose a new feature, it ran two half-weight single-feature ablations to isolate the cause. **Both still degraded performance.** The agent then declined to rescue the features and moved to honest negative-result reporting.

### Meta-learning — the "receipt"

In the final report, the agent synthesized a methodological lesson I had not written into the prompt:

> "Raw Pearson orthogonality does not imply portfolio complementarity. My pre-trade check found both features had near-zero correlation with mom_12_1 (0.07 and 0.008), which I read as a green light. It wasn't. Both features correlated **-0.28 / -0.35 with reversal_5d** — i.e., they encode the opposite sign of a working component at a horizon where that component dominates."

Then proposed a correction: proper orthogonalization via regressing the new feature on all existing features and using the residual. This is a well-known quant hygiene step; the agent re-derived it from the failure pattern.

### Follow-up suggestions (final report)

Three concrete ideas:
1. "Feature engineering that *assists* reversal rather than fights it" (sign-correct orthogonalization).
2. "Regime-aware weighting" — grounded in the agent's observation that IC-by-year varies ~5× across years (-1.2 IR in 2016, +2.1 IR in 2024).
3. "Weight sub-unit and sign-check against each active feature when correlations flip."

All grounded in specific numbers the agent pulled from tool responses.

---

## 4. Run 2 — knob-tuning without new features

**Goal**: "Without proposing any new features, find a weight-and-knob configuration that beats +0.128 net Sharpe. Run at least 2 configurations, diagnose each."
**Cost**: $0.69, 8 tool calls, 9 thinking blocks, 5 backtests across 5 distinct configs.

### Structured experiment design ✓
The agent tested 5 orthogonal directions: smoothing (hl=5, hl=4), exit-band widening, feature reweighting, signal-weighted sizing. Not random — each tested a specific hypothesis.

### Live self-correction on a tool semantics misunderstanding ✓✓
The most interesting moment of the entire evaluation. Agent tested `exit_n_deciles=7`, expected lower turnover, saw it unchanged:

> "Widening the exit threshold to 7 deciles actually hurt performance... the turnover stayed flat at 0.298, which doesn't make sense since a wider exit should reduce turnover. I'm realizing I may have misunderstood the parameter... If exit=5 means 'hold while in top 20%,' then exit=10 would mean holding almost everything, which would never exit. So smaller numbers might actually be the wider exit bands, not larger ones."

It updated its mental model of the parameter in real time based on empirical evidence, then correctly re-interpreted the result:

> "So my mental model needs flipping — higher exit_n_deciles actually means a *looser* threshold that holds worse names, not a tighter one... exit=7 tanked gross Sharpe from 0.37 to 0.15 without meaningfully reducing turnover, which suggests it's forcing the strategy to hold weaker-signal stocks that have drifted below the top decile."

This is a genuine debugging behavior — error signal → mental-model update → re-interpretation.

### Structural insights beyond the prompt ✓✓✓

The agent produced two findings I did not seed:

**Finding A: IC IR as a misleading metric.**
> "Across runs, IC IR stayed in the 1.08–1.18 band *even when gross Sharpe collapsed by 3×*. High IC captures the rank-linear signal, but the L/S book only trades the tails. Over-smoothing preserves IC while flattening decile spread... For decile strategies, track decile monotonicity, not IC."

**Finding B: mom_12_1's role.**
> "mom_12_1 is mostly dead weight for gross, valuable for stability. Doubling its weight halved turnover (30%→15%) and cost drag (1.9%→1.9%), but gross Sharpe went to ~0 because the signal became essentially equal-weighted long/short by 12-month momentum — which has near-zero cross-sectional edge in this universe over 2014–2026. It works in the baseline only as low-churn ballast that doesn't actively destroy the book."

These are findings I would expect from a careful human analyst who spent a few days on the pipeline.

### Final verdict

> "Best result: the existing baseline (+0.128 net Sharpe) remains unbeaten."

Ended with a table of 5 configs tested, plus a sharp distinction between "diagnostic failures" (taught us something real) and "just bad ideas" (worth checking once but dominated).

### Follow-up suggestions (final report)

1. **"The turnover problem is a feature problem, not a knob problem."** Proposed a slower-decay variant of the fast features (21-day vs 5-day reversal) as the right structural fix.
2. **Test the 3-feature subset** — drop mom_12_1 entirely given finding B.
3. **Regime diagnosis** — conditional weighting on dispersion regime.

---

## 5. Cross-run consistency — did the journal actually help?

Yes. Run 2's state recap included Run 1's history (`price_range_norm` registered, combined-feature runs recorded with their negative Sharpes). The agent:

- Did **not** re-propose overnight_drift_20d or signed_vol_20d.
- **Used** `price_range_norm` (from Haiku smoke test) as part of the baseline to beat.
- Referenced its own Run 1 diagnostics implicitly in reasoning about mom_12_1 and reversal dominance.

Cross-session memory worked as designed.

---

## 6. Honest caveats

1. **N=2 is not statistically meaningful.** Run-to-run consistency needs many more samples to validate. What this evaluation shows is "Opus *can* do this kind of work," not "Opus always does this kind of work."

2. **Feature proposals were in well-trodden territory.** Overnight-vs-intraday decomposition and accumulation/distribution are both published anomalies. The agent cited them correctly but didn't propose anything a moderately-well-read PM wouldn't recognize. A harder test: ask for features based on specific microstructure papers or a specific thesis.

3. **The agent's pre-trade forecasts were sometimes wrong** (predicted turnover 15–20% for `price_range_norm` in the Haiku smoke test, got 29.8%). It noted the miss but didn't build a correction mechanism for future predictions.

4. **No new profitable feature was found.** The agent was honest about this. Whether that's a limitation of Opus or a limitation of the existing feature basket (arguably saturated for price/volume-only daily data) is hard to separate.

5. **$1.20 is two runs on a tiny POC.** A full research rotation (say, a dozen runs covering different hypothesis classes) would be $10–$20. Still cheap, but worth noting that "cost of running Opus as a quant" is real.

---

## 7. Specific things to review with Yingyong

When you sit down together, the highest-signal artifacts to show him:

1. **Run 1's blowup-diagnosis-isolation sequence** (`outputs/research_20260419_220721/transcript.json`, blocks 3–10). The moment where it runs the combined backtest, reads the non-monotonic decile table, root-causes the sign-flip, then runs half-weight ablations to confirm. This is the strongest single demonstration of quant-flavored diagnostic thinking.

2. **Run 2's exit_n_deciles self-correction** (transcript.json, thinking block after the exit=7 backtest). The moment where it realizes it had the parameter semantics backwards, updates its mental model, and re-interprets the result. This is the closest thing to "the model catches its own mistake in real-time."

3. **Both final reports** (`final_report.md` in each run directory). Side-by-side, they illustrate the agent writing structured negative results — which is what PMs actually want to read, rather than hype.

4. **The meta-observation about IC vs decile-monotonicity** — arguably the single most valuable finding from both runs, and not one I had written into the system prompt.

---

## 8. What to push on next

1. **More hypothesis classes.** Price/volume is narrow. Worth testing the agent on: a) a specific microstructure question (e.g., "propose features from literature on closing auction imbalance"), b) feature-engineering from a custom thesis the PM provides, c) debugging a known-broken feature (as a control test for diagnostic ability).

2. **A self-calibration loop.** The agent predicts correlations and turnovers, then observes actuals, but doesn't track its own forecast errors. A tool like `record_prediction_vs_actual` that writes to the journal would let successive runs get better-calibrated.

3. **A "reviewer" role.** Run the same goal twice with different seeds/temperatures, then ask a third instance to compare the two and pick the better analysis. Tests whether the evaluation itself can be automated.

4. **Harder ground truth.** Right now the only success metric is net Sharpe on the local backtest. A stronger test: inject a synthetic feature with known properties (e.g., a look-ahead bug, or a feature that only works in 2020), see if the agent catches it via `analyze_last_run`.

5. **Budget discipline.** The agent sometimes burns thinking tokens in circles on decisions that are clear from data. Could tighten with `effort: medium` on sub-questions and `effort: max` reserved for final synthesis.
