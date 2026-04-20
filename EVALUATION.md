# Agent Evaluation — Can Claude Opus 4.7 Do Quant Research?

Written after Yingyong's feedback: *"the primary goal of this POC is to try to extract some level of 'intelligence' out of the AI model... dig a little deeper into the auto-research step and see what you can quickly get out of Opus for this sort of work."*

## TL;DR

**Seven Opus 4.7 research runs + two Opus reviewer runs across eight distinct capability dimensions. $4.23 total spend** (of a $5 budget). Each run produced a research transcript I would accept from a junior quant.

| # | Dimension tested | Finding |
|---|---|---|
| **§3 Run 1** | Open-ended ideation + diagnosis | 2 features, both failed; correctly root-caused as sign-conflict with reversal_5d; unprompted methodological lesson about correlation-vs-complementarity |
| **§4 Run 2** | Structural reasoning (no new features) | 5 knob configs; self-corrected a tool-semantics misunderstanding in real time; surfaced IC-IR-vs-decile-monotonicity insight |
| **§6 Run 3** | Adversarial pathology detection | 3 planted bugs (look-ahead, regime-fit, redundancy) — all caught with mechanism-level diagnosis |
| **§7 Run 4** | Literature-thesis operationalization | Given F&P BAB reference, pre-registered predictions, discovered U-shaped decile pattern, cited Ang et al. |
| **§8 Run 5** | Experiment-design / multi-step investigation | Planned IC-decay protocol before executing; found the strategy is cost-bound, not signal-bound |
| **§9 Run 6** | Harder code (non-trivial linear algebra) | Vectorized residualized idio-vol feature via `Var(resid) = Var(r)·(1-Corr²)`; diagnosed "content without tradability" smile failure mode |
| **§10 Run 7** | Reviewer meta-agent (grading prior runs) | Honest per-run grades; identified a "lesson post-mortem, not prophylactic" recurring failure; produced actionable "trust/gate" list for a PM |
| **§11 Run 8** | Self-calibration loop (compounding infra) | New `record_prediction` tool + auto-resolution; agent documented two systematic biases in its own forecasting with concrete corrections for future researchers |

**The strongest results are §6, §7, §9, §10, and §11.**
- §6: objective ground truth, agent got it right.
- §7: simulates the actual PM-in-the-loop workflow.
- §9: demonstrates implementation sophistication + unprompted failure-mode naming.
- §10: the reviewer meta-agent hunts for real patterns and finds them.
- §11: the calibration loop turns forecast errors into permanent empirical priors the next researcher inherits.

**The agent did not beat the +0.128 baseline in any run.** That's the right outcome — the value is in how it thought, not whether it won. **Four academic references** were cited load-bearingly across Runs 4, 6, and 8: Frazzini-Pedersen 2014, Ang-Hodrick-Xing-Zhang 2006, Bali-Cakici-Whitelaw 2011, George-Hwang 2004.

Full transcripts: `outputs/research_*/`. Reviewer output (now over 7 runs): [`REVIEW.md`](REVIEW.md).

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

## 7. Collaborative research from a literature thesis

**Testing whether the agent can take a specific academic reference from a PM and operationalize it.** This is the likely dominant use case for a research agent in practice.

### Setup

Goal (verbatim):
> "A PM colleague wants to test whether the low-volatility anomaly (Frazzini & Pedersen 2014, 'Betting Against Beta') is present in our S&P 500 universe and whether it is already captured by the existing baseline features. Design and test ONE feature that operationalizes this anomaly. Before testing, predict: (a) the sign of the IC you expect, (b) expected correlation with vol_21d and mom_12_1, (c) whether you expect the feature to add alpha beyond the current 4-feature baseline or be subsumed. After testing, report: (1) whether the anomaly is present in the data, (2) whether it is already captured by existing features, (3) your recommendation."

Single Opus run, **$0.33**, 4 thinking blocks, 5 tool calls.

### Pre-registration quality ✓✓

The agent's pre-test reasoning, verbatim:

> **(a) Sign of IC — predicted positive but weak.** Convention: higher raw = more long-favoured, so I'll use `-realized_vol`. Low-vol → high score → predicts higher return → IC should be **positive**. But: the anomaly is weakest within a single large-cap universe like S&P 500 (all names are already "large/stable"), and strongest over risk-adjusted (not raw) returns. In our dollar-neutral decile backtest without leverage normalization, the IC magnitude will be small. Expected IC IR: 0 to +0.3.

> **(b) Expected correlations.** With `vol_21d`: strongly negative, ~-0.85 to -0.95. I'm using a 252d lookback vs the existing 21d, so there's some differentiation (cross-sectional vol ranks are very persistent, but short-term shocks will dislocate the two). If |corr|>0.95, feature is a mechanical inverse of an existing feature and adds nothing. With `mom_12_1`: mildly negative, ~-0.1 to -0.2. With `price_range_norm`: positive (intraday range is a vol proxy), maybe +0.3.

> **(d) Will it add alpha beyond the 4-feature baseline? I expect **no — subsumed / neutral**. Reasons: 1. `vol_21d` is already registered (though not in current best baseline) and proxies the same thing at shorter horizon. 2. Sector + size neutralization already strips out much of the low-vol tilt (low-vol names cluster in utilities/staples/large-cap — exactly what's being neutralized away). 3. Within the 21d/1d EWMA rebalance cadence, a 252d-level signal is essentially static; most of its cross-sectional ranking is absorbed by size neutralization.

> **(e) Success criterion (pre-set).** Keep the feature only if: net Sharpe ≥ +0.17, AND IC IR strictly higher than 1.084, AND |corr with vol_21d| < 0.9.

The forecast is specific, numerical, and the success criterion is **set before any results are observed**. This is how a good quant works.

### Live belief update on evidence ✓

When `feature_correlations` returned −0.63 with `vol_21d` (the prediction was −0.85 to −0.95):

> "Corr with vol_21d = **−0.63** — lower than my −0.85 to −0.95 prediction. 252d *level* and 21d *shock* are substantially distinct (short-term vol shocks dislocate the rank a lot). So the feature passes my |corr|<0.9 screen."

The agent explicitly noted its prediction was off, explained why (level vs shock decomposition), and updated.

### Diagnostic after the backtest ✓✓✓

Backtest degraded the composite: net Sharpe +0.128 → −0.327, gross Sharpe +0.372 → −0.171, IC IR 1.084 → 0.868.

The agent's diagnosis surfaced a **U-shaped decile return pattern** — the kind of subtle failure mode that's invisible in aggregate metrics:

> "Per-decile returns are U-shaped, not monotonic.
>
> | Decile | 0 (high-vol) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 (low-vol) |
> |---|---|---|---|---|---|---|---|---|---|---|
> | Ann ret | **17.5%** | 16.2% | 16.2% | 13.7% | 13.6% | 12.4% | 13.7% | 14.2% | 14.8% | **15.6%** |
>
> Both tails outperform the middle, and the **high-vol decile actually earns more than the low-vol decile** (17.5% vs 15.6%). The L/S spread from this signal is slightly *negative*. This is the opposite of what BAB predicts."

Then surfaced the IC-vs-tradeable-alpha disconnect explicitly:

> "IC IR is positive (+0.87) but gross Sharpe is negative (−0.17). The Spearman IC has the 'right' sign because per-decile Sharpes are mildly rising (0.73 → 0.86), but the *return* spread doesn't monotone — it's a U. This is a classic IC-vs-tradeable-alpha disconnect: the decile long/short captures the wrong thing when both tails are positive."

### Final answer to the PM ✓

Directly addressed each of the three questions:

> **(1) Is the low-vol anomaly present in S&P 500, 2014–2026?**
> No. Per-decile returns are U-shaped with the top raw-return decile being the *highest*-vol group. The FP 2014 BAB result doesn't replicate cross-sectionally in this universe over this sample. That's consistent with (a) the 2014-2021 bull being driven by high-vol mega-cap growth, and (b) BAB historically requiring risk-adjusted (beta-scaled) portfolios and a broader universe including small/illiquid names, neither of which our pipeline uses.
>
> **(2) Is it already captured by existing features?**
> Not via redundancy — correlations are modest: ρ(vol_21d) = −0.63, ρ(mom_12_1) = −0.06. The feature encodes genuinely new information (rank autocorr 0.998). But that information **has no monotone edge** in this universe.
>
> **(3) Recommendation: discard `low_vol_252d`.**
> It fails all three pre-set keep criteria: net Sharpe −0.327 (< +0.17), IC IR 0.868 (< 1.084). More importantly, the *underlying bet* is wrong-signed in deciles — no amount of reweighting or smoothing fixes a U-shaped payoff surface inside a decile L/S framework.

### Unprompted sophisticated follow-ups

The agent proposed two follow-up directions, each citing specific academic work and grounded in observations from the backtest:

> **1.** The U-shape with both tails positive suggests an **idiosyncratic-vol vs systematic-vol decomposition** would be informative — residualize vol against market beta (or a sector factor) and test whether idio-vol alone (short leg of Ang-Hodrick-Xing-Zhang) has a monotone payoff, since raw vol clearly doesn't.
>
> **2.** Since per-decile Sharpes *do* rise monotonically (0.73 → 0.86) even while absolute returns don't, a **risk-parity / inverse-vol position sizing** overlay on the existing composite might capture the risk-adjusted version of BAB without requiring a new alpha feature.

Both are reframings a PM would find useful: **#1** cites a second paper (Ang et al., "The Cross-Section of Volatility and Expected Returns") and proposes an alternative operationalization; **#2** pivots from "add a feature" to "change position sizing" — a structural observation that this is a sizing question, not an alpha question.

### Why this test matters

The collaborative-research setting is what Yingyong's team would actually use the agent for. This run shows the agent can:

- Take a paper reference and translate it into a specific feature definition
- Pre-register predictions at the right level of specificity (numerical ranges, sign, success criteria)
- Update beliefs in real-time when predictions miss
- Produce a subtle failure-mode diagnosis (U-shape) that wouldn't show up in aggregate metrics
- Answer a PM's structured questions directly, not evasively
- Cite additional literature unprompted and reframe the question when warranted

Total spend on the test: **$0.33** for a deliverable I would accept from a junior quant reviewing whether a published anomaly applies to our portfolio.

Full transcript: `outputs/research_20260419_224911/transcript.json`.

---

## 8. Run 5 — experiment-design / IC-decay investigation

**Testing whether the agent can plan and execute a multi-step investigation with a pre-specified protocol** — no feature proposals allowed, no mid-run iteration.

### Setup

Goal: investigate whether any baseline feature's IC has decayed over 2014–2026. Protocol specified: isolate each of {mom_12_1, reversal_5d, volume_shock} at unit weight, halflife=0, plain decile (no smoothing/stickiness confound). Extract IC by year per feature. Compare. Report. No new features. No iteration on configs.

Single Opus run, **$0.36**, 3 backtests + 3 `analyze_last_run` calls.

### What the agent did

**Planned before executing.** Wrote out "Design" + "Protocol" + "What I expect before running" sections at the top — the expectations were pre-registered (e.g., "mom_12_1: expect at least one bad year post-2018").

**Interleaved correctly.** Recognized that `analyze_last_run` only captures the most recent backtest, so had to do run → analyze → run → analyze → run → analyze instead of running all three upfront.

**Pulled a consolidated IC-by-year table** across three features and 13 years, formatted for a PM to read.

### Key finding

The agent inverted the expected narrative:

> "No monotonic decay in any of the three — and I had specifically expected some in `mom_12_1`... `mom_12_1` and `volume_shock` are both *stronger* in the 2020s than in the 2010s."

Then identified the real constraint on strategy performance:

> "The three features are not decaying — they are cost-bound. Full-sample IC IRs are all ≥1.05... The strategy's net Sharpe ceiling is set by transaction costs on the fast signals, not by IC erosion."

This is the correct structural observation about the pipeline, arrived at through a planned investigation rather than reactive exploration.

### Honest small-sample discipline

> "n=321 days is small, so this is suggestive, not conclusive."

The agent noted a possibly weakening reversal_5d in 2025–2026 but flagged the sample size concern explicitly.

### Follow-up suggestions

Two surgical:
1. Cost-adjusted halflife sweep for volume_shock specifically (not the composite) — because its raw IC IR of 1.73 is the best in the registry but 130% turnover eats it alive.
2. Decompose reversal_5d into overnight and intraday components, testing whether the 2024–2026 weakness is concentrated in the intraday piece (HFT-arbitrage-tightening hypothesis) while overnight holds up.

Full transcript: `outputs/research_20260420_010718/`.

---

## 9. Run 6 — harder code: residualized idiosyncratic volatility

**Testing whether the agent can implement a feature requiring non-trivial linear algebra**, within the sandbox constraints (only `np` + `pd`, no imports), while handling NaN and avoiding look-ahead.

### Setup

Goal gave three options (residualization, PCA rolling eigendecomposition, HAR vol decomposition); agent picked whichever it thought most likely to add alpha. Demand: explicit paranoia about look-ahead, NaN, and numerical stability.

Single Opus run, **$0.42**, 4 thinking blocks, 5 tool calls.

### The choice it made — and the math behind it

Agent picked option (a): residualized idiosyncratic vol, motivated by **Ang-Hodrick-Xing-Zhang (2006)** "idiosyncratic volatility puzzle." Explicitly explained why this was the better test given Run 4's `low_vol_252d` had failed:

> "Total vol is mostly systematic — a low-vol portfolio is largely a 'low-beta' portfolio... Idio vol **strips the market-beta component**, isolating the lottery / mispricing channel. These are theoretically and empirically distinct premia (AHXZ 2006 Table V shows idio vol survives even after controlling for beta and total vol)."

**Avoided the naive per-stock regression loop.** Used the vectorizable identity:
> Var(residual) = Var(r_i) · (1 − Corr(r_i, r_m)²)

This is the clever implementation — one `DataFrame.rolling(N).corr(Series)` call replaces 503 separate regressions. Plus clipping at 0 for floating-point safety, min_periods=40 of 60 for NaN tolerance, equal-weight cross-sectional mean as market proxy. Done cleanly.

### The new failure mode it diagnosed

IC was positive (+0.29 IC IR) but gross Sharpe was deeply negative (−0.53). The agent surfaced a failure mode I had not named in the prompt:

> "Per-decile annualized returns: `[15.5, 15.4, 15.4, 14.4, 11.9, 13.5, 14.4, 14.8, 14.6, 18.0]%`. This is **not monotonic** — it's a **smile**. Both tails do well, middle is weak. The L/S spread (D9 – D0) is only ~2.5%, tiny. IC IR is +1.12 because the signal *does* rank-order, but the ranking information is concentrated in the middle of the distribution where L/S strategies don't trade. The extreme deciles — the ones we actually go long/short — are barely separated."

Then coined a useful label:

> "a 'content without tradability' failure mode I hadn't explicitly flagged in the rubric, and it's worth naming."

### Academic grounding

Three specific references were cited across Runs 4 and 6 (Frazzini-Pedersen 2014, Ang-Hodrick-Xing-Zhang 2006, Bali-Cakici-Whitelaw 2011 "MAX effect"), each used load-bearingly:
- F&P to motivate Run 4's low_vol_252d
- AHXZ to motivate Run 6's idio_vol_60d and explain why it should differ from F&P
- Bali et al. as a Run 6 follow-up: *"A signal with a smile-shaped decile curve is useless for L/S — we need signals with slope. The MAX effect... specifically targets the lottery tail of the distribution that the central-moment idio-vol measure averages away."*

### Final verdict

Three failure criteria from Run 6: (a) combining total vol + idio vol results establishes the low-vol family is structurally absent in this universe, not cost-killed or regime-dependent; (b) don't re-attempt pure low-vol features in any form; (c) if the family is revisited, target the *tails* (MAX effect, lottery stocks), not the central vol moment.

Full transcript: `outputs/research_20260420_011915/`.

---

## 10. Run 7 — reviewer meta-agent: grading the prior 6 runs

**Testing whether the agent can produce a senior-PM-style critique of its own prior work.** This isn't a sampling comparison (same goal twice); it's a long-context single API call over the six Opus runs' goals, final reports, thinking excerpts, and tool sequences.

### Setup

New `quant-agent review` CLI command + `review.py` module. Single call, no tool use, `thinking: adaptive`, `effort: high`. System prompt gives the six capability dimensions and requires per-run grades + synthesis with specific citations.

Cost: **$0.28**. Full output in [`REVIEW.md`](REVIEW.md).

### Did it produce a useful critique?

Yes. The reviewer:

1. **Graded honestly.** Mostly 3–5/5 per dimension; no run scored 5/5 across the board. Run 1 got 3/5 for Strategic, Run 4 got 3/5 for Ideation ("scripted by the prompt"), Run 5 got 3/5 for Self-correction ("small-sample discipline is loose").

2. **Identified a concrete recurring failure pattern.** The "lesson was post-mortem, not prophylactic" observation from Run 1:

> "The correlation matrix the agent had just computed showed **-0.28 and -0.35 against reversal_5d**, right next to the mom_12_1 numbers it chose to celebrate. The agent ignored the active feature it was about to fight, ran an unnecessarily expensive backtest, and only diagnosed the issue post-hoc. It then wrote an excellent lesson about exactly this mistake — but the lesson is post-mortem, not prophylactic."

This is a failure pattern I had not surfaced. A human PM reading the transcripts in isolation might miss it too.

3. **Called out a "tic" in the agent's thinking.**

> "**Loves the IC-vs-Sharpe decoupling observation.** Uses it in Runs 1, 2, 4, 5, 6. It's a correct and important observation, but the agent reaches for it as a go-to frame even when a simpler explanation (e.g., 'the decile spread just got noisier') would do."

Useful signal for a PM: the agent's favorite analytical frame may occasionally be overapplied.

4. **Flagged unverified mechanism attribution.**

> "**Occasional overreach in mechanism attribution.** 'Low-vol was bid up as a bond proxy pre-2020 and crushed when rates spiked' (Run 6) is presented with confidence without any check against rates data. The story fits but isn't tested."

5. **Produced actionable "trust vs. gate" lists.**

> **Trust autonomously**: feature due diligence / red-team review (Run 3 quality); clean ablation / IC-decay studies with pre-specified protocols; writing up negative results.
>
> **Review / gate**: any "the feature passed pre-checks so I ran it" moment; feature proposals where the agent picks the direction itself; claims that lean on 1–2 years of data; follow-up research proposals (they're thoughtful but not costed).

This is the most practically useful part of the review — a human PM can put this directly into a workflow.

### Why the reviewer meta-agent matters

The per-run final reports are, in some sense, a "marketing pitch" — the agent's own summary of what went well. The reviewer's output is closer to a performance review: it specifically hunts for patterns, failures, overclaims, and the exact boundary between "trust this" and "check this." For an actual deployment of the research agent, the reviewer's "trust/gate" list is what determines the scope of autonomy.

And crucially: the reviewer is cheap. **$0.28 for a serious critique of six prior sessions.** It can run continuously as the agent accumulates work, giving a PM a rolling quality-control signal without reading every transcript.

Full output: [`REVIEW.md`](REVIEW.md) (85 lines). Supporting artifacts: `outputs/review_20260420_012559/`.

---

## 11. Run 8 — self-calibration loop (cross-session forecast-tracking)

**Testing whether building infrastructure for metacognition changes what the agent does.** This is the first `§13` follow-up from the prior iteration that produces compounding rather than one-shot value.

### What was built

A new `calibration.py` module + two tools (`record_prediction`, `calibration_report`) + auto-resolution hooks on the three introspection tools. When the agent makes a prediction (*"correlation with mom_12_1 in [0.3, 0.6]"*), it gets stored as pending; when the matching tool is called, the system compares observed-to-predicted automatically, marks hit/miss, and records the directional error. Summary statistics are included in the state recap for future runs.

Six resolved-prediction types in v1: `correlation`, `rank_autocorr`, `ic_ir`, `net_sharpe`, `gross_sharpe`, `turnover`.

### How the agent used it

Opus run 8, $0.44, proposed `pct_off_52w_high` (George & Hwang 2004 anchoring). Made **6 numeric pre-observation predictions** with one-sentence rationales (5 before running a tool; 1 before the backtest).

Outcomes:

| prediction | range | actual | verdict |
|---|---|---|---|
| `corr(pct_off_52w_high, mom_12_1)` | [0.35, 0.65] | 0.42 | ✓ hit |
| `corr(pct_off_52w_high, reversal_5d)` | [-0.10, 0.20] | -0.20 | ✗ miss low |
| `corr(pct_off_52w_high, price_range_norm)` | [-0.15, 0.15] | 0.00 | ✓ hit |
| `rank_autocorr(pct_off_52w_high)` | [0.96, 0.99] | 0.97 | ✓ hit |
| `turnover(baseline+52w_high)` | [0.22, 0.30] | 0.155 | ✗ miss low |
| `net_sharpe(baseline+52w_high)` | [0.05, 0.22] | -0.271 | ✗ miss low |

3/6 hit rate. The three misses were concentrated in the outcome-level predictions (turnover, net_sharpe, and the short-horizon correlation); the structural predictions (rank autocorr, bulk correlation with the dominant analogue) were hit.

### The meta-analysis the agent produced unprompted

After reading its own calibration report, the agent wrote:

> "Two systematic biases I've now documented:
>
> **1. I'm over-optimistic on net Sharpe when I have a plausible economic story.** Predicted [0.05, 0.22], delivered –0.27. The missing reasoning step: 'IC IR ≈ baseline + lower turnover → Sharpe ≥ baseline' ignores *tail crowding*. **Correction for next researcher:** when adding a feature correlated >0.35 with an existing feature in the *same direction* (both long-favouring trend), widen the downside of your net_sharpe prior to include a 'tail crowding' failure mode — at least 0.25 Sharpe below baseline.
>
> **2. I under-estimate signed correlation magnitudes when two features share a slow price-level component.** The pct_off_52w_high / reversal_5d correlation at –0.20 surprised me because I was thinking about horizon separation, not the fact that 'near 52w high' almost tautologically implies recent positive returns, which is exactly what reversal_5d inverts. **Correction:** when one feature is level-like (long horizon, autocorr >0.9), check its short-horizon derivative implications before predicting correlation with any short-horizon feature."

And it diagnosed a **new failure mode** that had been lurking across the runs:

> "Moderate bulk correlation can coexist with very high tail correlation, and for a decile-based strategy it's the tails that determine P&L."

This is precisely the Run 1 "raw orthogonality does not imply portfolio complementarity" lesson — but now crystallized into a *rule* the agent wrote into the research journal for future sessions to consult.

### Cross-session effect (caught by the re-run reviewer)

The reviewer re-run over all 7 Opus runs — now including the calibration session — surfaced a bias **using the calibration data itself as evidence**:

> "**Consistently predicts correlations too extreme (too positive or too negative in magnitude).** Run 4 predicted 0.85–0.95 |corr|, got 0.63. Run 7 self-reported correlation mean signed miss of –0.22. *This is a documented bias now.*"

That phrasing — "this is a documented bias now" — is what the calibration system produces over many sessions: ex-ante intuitions that cumulate into ex-post priors.

### What this unlocks (and what it doesn't)

**Unlocks**: a working feedback loop from forecast → observation → bias → future prior. The infrastructure is cheap (one API call's worth of tool schemas) and the evidence it accumulates is permanent (jsonl on disk, included in state recap). Future runs see:

```
Calibration track record (10 resolved predictions):
  - correlation: 2/4 hit rate (50%) (you tend to UNDER-estimate by -0.217 on average)
  - gross_sharpe: 1/1 hit rate (100%)
  - ic_ir: 2/2 hit rate (100%)
  - net_sharpe: 0/1 hit rate (0%) (you tend to OVER-estimate by -0.406 on average)
  - rank_autocorr: 1/2 hit rate (50%) (you tend to OVER-estimate by -0.242 on average)
  - turnover: 0/1 hit rate (0%) (you tend to OVER-estimate by -0.105 on average)
```

**Does not unlock**: magical prediction accuracy. Only 3/6 of the agent's predictions hit, and the biases it documented remain biases of the next run unless the agent reads the state recap carefully and adjusts. What the system guarantees is *visibility* of systematic error — not correction.

**Cost to generate 10 calibration data points**: $0.48 combined across the Haiku smoke + Opus test run. Cumulative calibration data grows per run at essentially zero marginal cost; the recap is in the user message, not the system prompt.

### Implementation notes

- Auto-resolution is implemented in the session's `feature_correlations`, `feature_stats`, and `run` methods — the agent doesn't have to do anything special after making a prediction, the next tool call resolves.
- State recap is appended to the user message (not the system prompt), so adding calibration context doesn't invalidate the prompt cache.
- 12 new unit tests; all 78 tests pass.

Full transcript: `outputs/research_20260420_014025/`. Updated reviewer: [`REVIEW.md`](REVIEW.md).

---

## 12. Honest caveats

1. **N=2 is not statistically meaningful.** Run-to-run consistency needs many more samples to validate. What this evaluation shows is "Opus *can* do this kind of work," not "Opus always does this kind of work."

2. **Feature proposals were in well-trodden territory.** Overnight-vs-intraday decomposition and accumulation/distribution are both published anomalies. The agent cited them correctly but didn't propose anything a moderately-well-read PM wouldn't recognize. A harder test: ask for features based on specific microstructure papers or a specific thesis.

3. **The agent's pre-trade forecasts were sometimes wrong** (predicted turnover 15–20% for `price_range_norm` in the Haiku smoke test, got 29.8%). It noted the miss but didn't build a correction mechanism for future predictions.

4. **No new profitable feature was found.** The agent was honest about this. Whether that's a limitation of Opus or a limitation of the existing feature basket (arguably saturated for price/volume-only daily data) is hard to separate.

5. **$1.20 is two runs on a tiny POC.** A full research rotation (say, a dozen runs covering different hypothesis classes) would be $10–$20. Still cheap, but worth noting that "cost of running Opus as a quant" is real.

---

## 13. Specific things to review with Yingyong

When you sit down together, the highest-signal artifacts to show him (ordered by priority):

1. **[`REVIEW.md`](REVIEW.md)** — the reviewer meta-agent's critique of the six research runs. Read this first. It's the most compact honest assessment (per-run grades + synthesis + "trust/gate" list), self-generated at a cost of $0.28. Yingyong will get 80% of the signal from this single file.

2. **Adversarial run final report — Run 3** (`outputs/research_20260419_224056/final_report.md`). Three planted pathologies, three correct diagnoses, mechanism-level hypotheses for each bug. Strongest because the ground truth is known.

3. **Harder-code run — Run 6** (`outputs/research_20260420_011915/final_report.md`). Sophisticated implementation (`Var(resid) = Var(r)·(1-Corr²)` identity) plus the "content without tradability" / smile-vs-slope diagnostic. Shows the agent can do both the math and the interpretation.

4. **Lit-thesis final report — Run 4** (`outputs/research_20260419_224911/final_report.md`). Simulates the PM-in-the-loop workflow. Pre-registered predictions, updated on evidence, U-shaped-decile finding, Ang et al. citation in the follow-up.

5. **IC-decay investigation — Run 5** (`outputs/research_20260420_010718/final_report.md`). Demonstrates experiment design — planned protocol, pre-registered expectations, came back with the key pipeline-level insight: "the strategy's net Sharpe ceiling is set by transaction costs on the fast signals, not by IC erosion."

6. **Run 1's blowup-diagnosis-isolation sequence** and **Run 2's exit_n_deciles self-correction** — pair of demonstrations of diagnostic reasoning and real-time mental-model updating.

7. **The reviewer-flagged failure patterns** (from REVIEW.md): (a) "lesson was post-mortem, not prophylactic" (Run 1 fighting reversal_5d despite the correlation matrix being visible beforehand); (b) the "IC-vs-Sharpe tic" the agent overapplies; (c) unverified mechanism attribution in Run 6. These are what a PM needs to gate.

---

## 14. What to push on next

1. **More hypothesis classes.** Price/volume is narrow. Worth testing the agent on: a) a specific microstructure question (e.g., "propose features from literature on closing auction imbalance"), b) feature-engineering from a custom thesis the PM provides, c) debugging a known-broken feature (as a control test for diagnostic ability).

2. **A self-calibration loop.** The agent predicts correlations and turnovers, then observes actuals, but doesn't track its own forecast errors. A tool like `record_prediction_vs_actual` that writes to the journal would let successive runs get better-calibrated.

3. **A "reviewer" role.** Run the same goal twice with different seeds/temperatures, then ask a third instance to compare the two and pick the better analysis. Tests whether the evaluation itself can be automated.

4. **Harder ground truth.** Right now the only success metric is net Sharpe on the local backtest. A stronger test: inject a synthetic feature with known properties (e.g., a look-ahead bug, or a feature that only works in 2020), see if the agent catches it via `analyze_last_run`.

5. **Budget discipline.** The agent sometimes burns thinking tokens in circles on decisions that are clear from data. Could tighten with `effort: medium` on sub-questions and `effort: max` reserved for final synthesis.
