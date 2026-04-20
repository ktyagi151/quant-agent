# Agent Evaluation — Can Claude Opus 4.7 Do Quant Research?

Written after Yingyong's feedback: *"the primary goal of this POC is to try to extract some level of 'intelligence' out of the AI model... dig a little deeper into the auto-research step and see what you can quickly get out of Opus for this sort of work."*

## TL;DR

Four Opus 4.7 runs across four distinct capability dimensions. **$2.37 total.** Each produced a research transcript I would accept from a junior quant.

| # | Dimension tested | Finding |
|---|---|---|
| **§3 Run 1** | Open-ended ideation + diagnosis | Proposed 2 features, both failed; correctly root-caused failure as a sign-conflict with reversal_5d; surfaced unprompted methodological lesson about correlation-vs-complementarity |
| **§4 Run 2** | Structural reasoning (no new features) | Tested 5 distinct knob configurations; self-corrected a tool-semantics misunderstanding in real time; surfaced the IC-IR-vs-decile-monotonicity insight |
| **§6 Run 3** | Adversarial pathology detection | Planted 3 features with hidden bugs (look-ahead, regime-fit, redundancy); agent caught all 3 with mechanism-level diagnosis |
| **§7 Run 4** | Collaborative research from literature | Given Frazzini-Pedersen BAB reference, operationalized it, pre-registered predictions, discovered U-shaped decile pattern (non-monotonic), cited Ang et al. in follow-up |

**The strongest results are §6 and §7.** §6 because the ground truth is known and the agent got it right; §7 because it simulates the actual PM-in-the-loop workflow and produced a subtle-failure-mode diagnosis (U-shape in deciles) that aggregate metrics would have missed.

**The agent did not beat the +0.128 baseline in any run.** That's the right outcome — the value is in how it thought, not whether it won.

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

## 8. Honest caveats

1. **N=2 is not statistically meaningful.** Run-to-run consistency needs many more samples to validate. What this evaluation shows is "Opus *can* do this kind of work," not "Opus always does this kind of work."

2. **Feature proposals were in well-trodden territory.** Overnight-vs-intraday decomposition and accumulation/distribution are both published anomalies. The agent cited them correctly but didn't propose anything a moderately-well-read PM wouldn't recognize. A harder test: ask for features based on specific microstructure papers or a specific thesis.

3. **The agent's pre-trade forecasts were sometimes wrong** (predicted turnover 15–20% for `price_range_norm` in the Haiku smoke test, got 29.8%). It noted the miss but didn't build a correction mechanism for future predictions.

4. **No new profitable feature was found.** The agent was honest about this. Whether that's a limitation of Opus or a limitation of the existing feature basket (arguably saturated for price/volume-only daily data) is hard to separate.

5. **$1.20 is two runs on a tiny POC.** A full research rotation (say, a dozen runs covering different hypothesis classes) would be $10–$20. Still cheap, but worth noting that "cost of running Opus as a quant" is real.

---

## 9. Specific things to review with Yingyong

When you sit down together, the highest-signal artifacts to show him (ordered by priority):

1. **Adversarial run final report — Run 3** (`outputs/research_20260419_224056/final_report.md`). Three planted pathologies, three correct diagnoses, mechanism-level hypotheses for each bug. The strongest single artifact because the ground truth is known and the agent got it right.

2. **Lit-thesis final report — Run 4** (`outputs/research_20260419_224911/final_report.md`). Shows the agent executing the workflow Yingyong's team would actually use: PM gives a paper reference, agent operationalizes, tests, reports. Produced the U-shaped-decile diagnostic insight that aggregate metrics would have missed, and cited a second paper (Ang et al.) in the follow-up.

3. **Run 1's blowup-diagnosis-isolation sequence** (`outputs/research_20260419_220721/transcript.json`, blocks 3–10). The combined backtest tanks, agent reads the non-monotonic decile table, root-causes the sign-flip, runs half-weight ablations to confirm. Clean demonstration of quant-flavored diagnostic thinking.

4. **Run 2's exit_n_deciles self-correction** (`outputs/research_20260419_221227/transcript.json`, thinking block after the exit=7 backtest). The moment where it realizes it had the parameter semantics backwards, updates its mental model, re-interprets the result. Closest thing to "the model catches its own mistake in real-time."

5. **The recurring IC-IR-vs-decile-monotonicity observation** — surfaced unprompted in Runs 2, 3, and 4. This is the closest thing to an emergent "lesson" the agent developed across runs. The journal's state-recap mechanism is partly responsible (it saw the prior runs' findings and built on them).

---

## 10. What to push on next

1. **More hypothesis classes.** Price/volume is narrow. Worth testing the agent on: a) a specific microstructure question (e.g., "propose features from literature on closing auction imbalance"), b) feature-engineering from a custom thesis the PM provides, c) debugging a known-broken feature (as a control test for diagnostic ability).

2. **A self-calibration loop.** The agent predicts correlations and turnovers, then observes actuals, but doesn't track its own forecast errors. A tool like `record_prediction_vs_actual` that writes to the journal would let successive runs get better-calibrated.

3. **A "reviewer" role.** Run the same goal twice with different seeds/temperatures, then ask a third instance to compare the two and pick the better analysis. Tests whether the evaluation itself can be automated.

4. **Harder ground truth.** Right now the only success metric is net Sharpe on the local backtest. A stronger test: inject a synthetic feature with known properties (e.g., a look-ahead bug, or a feature that only works in 2020), see if the agent catches it via `analyze_last_run`.

5. **Budget discipline.** The agent sometimes burns thinking tokens in circles on decisions that are clear from data. Could tighten with `effort: medium` on sub-questions and `effort: max` reserved for final synthesis.
