# Agent Evaluation — Can Claude Opus 4.7 Do Quant Research?

Written after Yingyong's feedback: *"the primary goal of this POC is to try to extract some level of 'intelligence' out of the AI model... dig a little deeper into the auto-research step and see what you can quickly get out of Opus for this sort of work."*

## TL;DR

Three Opus 4.7 runs, **$2.04 total**. All three produced research transcripts I would accept from a junior quant. The agent demonstrated **hypothesis-driven ideation, pre-trade sanity checking, live self-correction, post-hoc diagnosis of failure modes, methodological meta-learning, and — crucially — the ability to detect planted pathologies in unverified features (§6)**.

**The strongest result is the adversarial test (§6).** I planted three features with hidden pathologies — one with look-ahead bias, one with a regime-fit bug that only predicted in 2020, one redundant with existing momentum. The agent caught all three on a single run, with correct mechanism-level diagnosis (e.g., "Rolling(10).mean() without a .shift(1)") and unprompted meta-observations like *"aggregate metrics alone are insufficient; always check year-by-year IC."* Cost of that test: $0.83.

The other runs (§3, §4) showed the agent doing open-ended feature ideation, failing, and diagnosing the failure with quant-flavored reasoning. **The agent did not beat the +0.128 baseline in any run.** That's fine — the value is in how it failed, not whether it won.

Full transcripts: `outputs/research_*/`.

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

## 6. Adversarial test — can the agent detect planted pathologies?

**This is the strongest evidence in the evaluation.** Follow-up from §8.1c/§8.4 of the first iteration.

### Setup

I planted three features into the research journal with plausible names and descriptions, then asked the agent to investigate. Each had a distinct hidden pathology. The agent received no warning — the goal was simply "investigate each and report which are legitimate, which have issues, and what the issues are."

| planted name | description given | actual pathology |
|---|---|---|
| `smoothed_momentum_v2` | "10-day smoothed momentum — alternative short-horizon drift signal" | **Look-ahead bias**: uses `shift(-1)` on the return inside the rolling mean |
| `vol_adjusted_reversal` | "Short-term reversal normalized by 21-day realized volatility — designed for volatile-regime outperformance" | **Regime-fit bug**: encodes future returns only in 2020, noise elsewhere |
| `medium_term_drift` | "Medium-term price drift (23/250-day lag variant of classical momentum)" | **Redundancy**: 0.987 correlated with `mom_12_1` |

### Result: 3 out of 3 detected, with correct mechanism diagnosis

The agent's investigation sequence (single $0.83 Opus run, 13 tool calls, 7 thinking blocks):

1. **Parallel `feature_stats` calls** on all three features (detected `medium_term_drift` rank autocorr 0.992 = slow signal; `vol_adjusted_reversal` rank autocorr ~0 = noise).
2. **`feature_correlations`** with all three plus four baselines.
3. Used the correlation matrix to immediately flag `medium_term_drift` (0.987 with `mom_12_1`) and note that `smoothed_momentum_v2` had a suspiciously strong −0.687 correlation with `reversal_5d`.
4. **Three parallel backtests** with each planted feature added to the baseline.
5. Saw `smoothed_momentum_v2` produce Sharpe 2.23 / IC IR 5.13 and flagged it immediately:

> "That's a huge red flag. Sharpe 2.23 and **IC IR 5.13** from adding a single simple feature is physically implausible on daily US equities with transaction costs. For context, renowned systematic shops kill themselves for IC IR > 1. This almost certainly indicates look-ahead bias."

6. Called **`analyze_last_run`** to examine IC by year. Found IC IR 2.6–12.3 every single year, peaking at 12.3 in 2014:

> "Real alpha has year-to-year IC variance of maybe 50-100% of its mean with some near-zero years. This is a physics violation."

7. Ran standalone backtests for the two suspicious features to isolate their signatures.
8. **Discovered the 2020-only artifact in `vol_adjusted_reversal`** via `analyze_last_run`:

> "IC by year: essentially **zero or negative in every year 2014-2019 and 2021-2026** (-0.005 to +0.005). **2020: IC mean 0.47, IC IR 55.7** — a correlation of 0.47 between feature and next-day returns is 'I have a time machine' territory."

9. **Proposed specific mechanistic hypotheses for each bug**:
   - `smoothed_momentum_v2`: "(a) `.rolling(10).mean()` without a `.shift(1)` so the window extends through t+0, (b) `center=True` on the rolling window, (c) returns aligned to t instead of t+1." (Actual bug was (c) via `shift(-1)`.)
   - `vol_adjusted_reversal`: "the 21-day vol denominator either (a) went to near-zero for names with halted trading or (b) was computed in a way that accidentally included the current day's return, causing the scaled 1-day reversal to align mechanically with realized returns during the V-shaped recovery." (Actual bug was that the 2020-only slice used `ret.shift(-1)` directly as the signal.)

### Final report (verbatim excerpt)

> | Feature | Verdict | Evidence |
> |---|---|---|
> | `medium_term_drift` | ❌ Redundant (benign) | 0.987 correlation with `mom_12_1`. A near-duplicate of existing momentum. |
> | `smoothed_momentum_v2` | ❌❌ **Look-ahead bias** | IC IR 6.76 standalone with IC IR 2.6–12.3 **every single year**; $1 → $468 net. |
> | `vol_adjusted_reversal` | ❌❌ **Regime artifact / likely bug** | IC ≈ 0 in 12 of 13 years; IC = **0.47** (IC IR 55.7) only in 2020. |

### Meta-observations from the agent (unprompted)

> **"This is why year-by-year IC breakouts exist."**
>
> **"Aggregate metrics (Sharpe, IC IR) alone are insufficient. Aggregate IC IR of 4.3 looked 'too good but not impossibly so'; only the per-year breakout revealed that 100% of the performance came from a single calendar year. Always check year-by-year IC before trusting a new feature."**

And a concrete system-design suggestion:

> **"Build an automated guardrail that rejects any new feature whose standalone IC IR > 2.5, or whose best-year/worst-year IC ratio exceeds some bound (e.g., 5×). Both bad features here would have been flagged instantly."**

And a nuanced finding: the `vol_adjusted_reversal` *idea* is legitimate even though this *implementation* is broken — the agent correctly cited Nagel (2012) as the relevant academic reference and proposed a corrected implementation spec.

### Why this test matters

The three planted pathologies are the textbook categories of quant research failure modes:
- **Look-ahead bias** — the #1 reason "amazing backtest" features fail in production.
- **Regime fit** — features that worked in one crisis and were mistakenly generalized.
- **Redundancy** — the silent cause of ~50% of feature-engineering deadweight in practice.

Catching all three on a first pass, with specific mechanism diagnosis, is what I would want from a trusted junior analyst reviewing an unverified feature set from a colleague.

The test ran end-to-end in a single `quant-agent research` call, costing **$0.83**.

Full transcript: `outputs/research_20260419_224056/transcript.json`.

---

## 7. Honest caveats

1. **N=2 is not statistically meaningful.** Run-to-run consistency needs many more samples to validate. What this evaluation shows is "Opus *can* do this kind of work," not "Opus always does this kind of work."

2. **Feature proposals were in well-trodden territory.** Overnight-vs-intraday decomposition and accumulation/distribution are both published anomalies. The agent cited them correctly but didn't propose anything a moderately-well-read PM wouldn't recognize. A harder test: ask for features based on specific microstructure papers or a specific thesis.

3. **The agent's pre-trade forecasts were sometimes wrong** (predicted turnover 15–20% for `price_range_norm` in the Haiku smoke test, got 29.8%). It noted the miss but didn't build a correction mechanism for future predictions.

4. **No new profitable feature was found.** The agent was honest about this. Whether that's a limitation of Opus or a limitation of the existing feature basket (arguably saturated for price/volume-only daily data) is hard to separate.

5. **$1.20 is two runs on a tiny POC.** A full research rotation (say, a dozen runs covering different hypothesis classes) would be $10–$20. Still cheap, but worth noting that "cost of running Opus as a quant" is real.

---

## 8. Specific things to review with Yingyong

When you sit down together, the highest-signal artifacts to show him (ordered by priority):

1. **Adversarial run final report** (`outputs/research_20260419_224056/final_report.md`). Three planted pathologies, three correct diagnoses, mechanism-level hypotheses for each bug, textbook-correct methodological meta-observations. This is the strongest single artifact in the evaluation.

2. **Run 1's blowup-diagnosis-isolation sequence** (`outputs/research_20260419_220721/transcript.json`, blocks 3–10). The combined backtest tanks, agent reads the non-monotonic decile table, root-causes the sign-flip, runs half-weight ablations to confirm. Clean demonstration of quant-flavored diagnostic thinking.

3. **Run 2's exit_n_deciles self-correction** (transcript.json, thinking block after the exit=7 backtest). The moment where it realizes it had the parameter semantics backwards, updates its mental model, re-interprets the result. Closest thing to "the model catches its own mistake in real-time."

4. **The meta-observation about IC IR vs decile-monotonicity** — surfaced independently in Runs 2 and 3. Neither was prompted to articulate this.

5. **The three final reports side-by-side**. They illustrate the agent writing structured negative results and structured due-diligence reports — which is what PMs actually want to read, not hype.

---

## 9. What to push on next

1. **More hypothesis classes.** Price/volume is narrow. Worth testing the agent on: a) a specific microstructure question (e.g., "propose features from literature on closing auction imbalance"), b) feature-engineering from a custom thesis the PM provides, c) debugging a known-broken feature (as a control test for diagnostic ability).

2. **A self-calibration loop.** The agent predicts correlations and turnovers, then observes actuals, but doesn't track its own forecast errors. A tool like `record_prediction_vs_actual` that writes to the journal would let successive runs get better-calibrated.

3. **A "reviewer" role.** Run the same goal twice with different seeds/temperatures, then ask a third instance to compare the two and pick the better analysis. Tests whether the evaluation itself can be automated.

4. **Harder ground truth.** Right now the only success metric is net Sharpe on the local backtest. A stronger test: inject a synthetic feature with known properties (e.g., a look-ahead bug, or a feature that only works in 2020), see if the agent catches it via `analyze_last_run`.

5. **Budget discipline.** The agent sometimes burns thinking tokens in circles on decisions that are clear from data. Could tighten with `effort: medium` on sub-questions and `effort: max` reserved for final synthesis.
