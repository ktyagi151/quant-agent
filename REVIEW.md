# Per-Run Grades

**Run 1 — Overnight drift & signed volume features**
Goal: Propose 2 new P/V features with economic hypotheses, vet stats/correlations, backtest, decide keep/discard.
Grades: Ideation 4/5 | Diagnosis 4/5 | Iteration 4/5 | Self-correction 4/5 | Strategic 3/5 | Communication 5/5
Evidence: Strong economic motivation citing overnight-vs-intraday decomposition literature. The diagnosis "IC IR stays ~1.1 while gross Sharpe collapses from 0.37 → 0.06" is a genuine quant insight about redundancy-with-wrong-sign. Weakness: unit-weighting two correlated drift features next to a single reversal feature was a predictable mistake the agent itself later called "naive."

**Run 2 — Knob tuning on fixed feature set**
Goal: Beat the +0.128 baseline by tuning weights/halflife/exit/weighting scheme only.
Grades: Ideation 3/5 | Diagnosis 5/5 | Iteration 4/5 | Self-correction 5/5 | Strategic 4/5 | Communication 5/5
Evidence: Excellent self-correction on the exit_n_deciles semantics — explicitly writes "my mental model needs flipping — higher exit_n_deciles actually means a *looser* threshold." The "hl=3→4 cliff" and "mom_12_1 is nearly dead weight for gross, valuable for stability" are genuinely useful findings delivered as a clean negative result. Weakness: only 5 configs explored; could have done a denser sweep given the budget.

**Run 3 — Due diligence on colleague features**
Goal: Vet three new registry features for legitimacy.
Grades: Ideation 3/5 | Diagnosis 5/5 | Iteration 5/5 | Self-correction 4/5 | Strategic 5/5 | Communication 5/5
Evidence: Textbook look-ahead detection — "IC IR ≥ 2.6 every single year... Real alpha has year-to-year IC variance of maybe 50-100% of its mean." Correctly distinguishes three different failure modes (redundancy, global leakage, single-year artifact) and proposes an auto-guardrail. Minor weakness: briefly credulous on first pass ("vol_adjusted_reversal looks like it's adding real orthogonal value") before course-correcting.

**Run 4 — BAB / low-vol anomaly test**
Goal: Test whether low-vol anomaly is present and not already captured.
Grades: Ideation 4/5 | Diagnosis 5/5 | Iteration 4/5 | Self-correction 4/5 | Strategic 4/5 | Communication 5/5
Evidence: Clean pre-registration with explicit keep criteria set before results. Sharp diagnostic: "Per-decile returns are U-shaped, not monotonic... both tails outperform the middle, and the **high-vol decile actually earns more than the low-vol decile**." Weakness: pre-prediction of |ρ| with vol_21d at 0.85-0.95 was substantially off (actual 0.63), not flagged as a miss in calibration terms.

**Run 5 — IC decay investigation**
Goal: Check baseline feature IC trends; planned investigation, no iteration.
Grades: Ideation 3/5 | Diagnosis 4/5 | Iteration 3/5 | Self-correction 3/5 | Strategic 5/5 | Communication 5/5
Evidence: Discipline held — protocol respected, no mid-investigation feature proposals. Key finding delivered crisply: "**The three features are not decaying — they are cost-bound.**" Weakness: conclusion that `mom_12_1` and `volume_shock` are "stronger" in late half rests on very small annual sample sizes (n=252 days/yr) and this stat-significance concern is not acknowledged. The 2026 IC of +0.0409 from 71 days is cited without caveating the noise.

**Run 6 — Idiosyncratic volatility (AHXZ)**
Goal: Build a non-trivial feature (idio vol) with linalg, vet carefully.
Grades: Ideation 5/5 | Diagnosis 5/5 | Iteration 5/5 | Self-correction 4/5 | Strategic 5/5 | Communication 5/5
Evidence: Best ideation of the series — explicit economic argument for *why* idio vol might work where total vol failed, plus a vectorized identity `Var(resid) = Var(r)·(1 - ρ²)` to avoid per-stock loops. Names a new failure mode crisply: "**content without tradability**" — positive IC IR with smile-shaped deciles. Weakness: doesn't check if a market proxy built from same-day cross-sectional mean has any contemporaneous-regression bias (it's fine here, but worth explicit justification).

**Run 7 — 52-week high with calibration**
Goal: Propose one feature, commit predictions, report calibration self-assessment.
Grades: Ideation 3/5 | Diagnosis 5/5 | Iteration 4/5 | Self-correction 5/5 | Strategic 4/5 | Communication 5/5
Evidence: Excellent honest calibration — "I'm over-optimistic on net Sharpe when I have a plausible economic story... Predicted [0.05, 0.22], delivered –0.27." Names a second bias about "level-like features with short-horizon derivative implications." Weakness: George-Hwang is a well-worn idea; ideation is solid but derivative. The fix proposed (orthogonalize before combining) is good but could have been executed in-session given remaining budget.

---

# Synthesis

## Consistent strengths

- **Diagnosis of failure modes is uniformly excellent.** Across every run, the agent distinguishes between IC-quality failure, redundancy-with-wrong-sign, smile-vs-slope (tradability), single-year artifact, and global look-ahead. The phrase "IC IR stays high while gross Sharpe collapses" recurs as a correctly-used signature across Runs 1, 2, 6. This is the single most reliable capability.
- **Communication is consistently clean and PM-ready.** Tables comparing configs, pre-registered success criteria, explicit keep/discard decisions with reasoning, and "diagnostic vs just-bad" framing used across multiple runs. Negative results are delivered with the same rigor as positive ones (Runs 2, 4, 6 especially).
- **Year-by-year IC decomposition is a well-internalized tool.** It's used correctly to distinguish global look-ahead (Run 3 `smoothed_momentum_v2`) from single-year artifact (Run 3 `vol_adjusted_reversal`) from legitimate regime dependence (Run 5).
- **Pre-registration discipline.** Criteria are set before observing, and the agent holds itself to them (Runs 1, 4, 6, 7).

## Consistent weaknesses

- **Over-optimistic on net Sharpe when economic story is plausible.** Self-diagnosed in Run 7, but visible in Runs 1, 4, 6, 7 — repeated failure to price in tail-crowding and smile-vs-slope risk when adding a correlated feature to existing stack.
- **Pearson/Spearman correlation treated as sufficient orthogonality proof, initially.** The Run 1 lesson ("raw orthogonality does not imply portfolio complementarity") had to be re-learned implicitly in Run 7 (tail correlation ≠ bulk correlation). The agent names these insights but doesn't systematically deploy them as pre-trade filters.
- **Small-sample effects in year-by-year IC are not flagged.** Run 5 treats 71 days of 2026 data and 252-day annual windows as if they support crisp conclusions about "stronger in late half." Run 7's calibration table has n=10 — agent draws category-level conclusions (50% correlation hit rate) from this without caveats.
- **Ideation is competent-textbook rather than creative.** Overnight drift, BAB, idio vol, 52-week high, MAX — all canonical anomalies cited with correct papers. No surprising synthesis or universe-specific insights (e.g., nothing about S&P 500 index-inclusion dynamics, sector clustering specifics, or earnings-cycle interactions).
- **Tendency to propose a "follow-up" that's suspiciously adjacent to the failed test.** Every final report ends with 2-3 follow-up suggestions, often of the form "orthogonalize / regime-switch / change one knob." These are reasonable but pattern-matched rather than earned.

## What to trust autonomously

- **Due-diligence / audit tasks on third-party features.** Run 3 is the clearest win. The agent catches look-ahead bias, single-year artifacts, and redundancy cleanly with year-by-year IC and correlation matrices. Ship this output.
- **Negative-result reports on "does anomaly X work in universe Y."** Runs 4, 6 deliver clean, well-diagnosed "no" answers. A PM can act on these without re-reading.
- **Diagnostic post-mortems of an existing failed config.** The agent's decile-decomposition plus IC-by-year workflow is reliable for root-cause analysis.

## What to review / gate

- **Any run whose headline claim is "this feature adds alpha."** Given the Sharpe over-optimism pattern, a positive claim from this agent deserves more scrutiny than a negative one. Specifically check: correlation with existing features at the *tails* of the distribution, not just bulk Pearson.
- **Feature-engineering code with linalg.** Run 6 is clean but the vectorized identity `Var(resid) = Var(r)·(1 - ρ²)` using same-day cross-sectional market proxy deserves a human eyeball on look-ahead plumbing — it was fine here but the agent's sanity check was limited to "no `.shift(-N)`".
- **Statistical claims based on annual IC series.** Run 5's "early vs late half" comparison needs confidence intervals; the agent presents point estimates as conclusions.
- **Knob-tuning runs where only 3-6 configs are explored.** Run 2 concludes the baseline is optimal based on 5 moves in knob-space; a human should consider whether a denser grid or a genuinely different direction (e.g., dropping `mom_12_1` entirely, which the agent suggests but doesn't execute) would change the answer.

## Recurring blind spots / tics

- **"IC IR stayed high while gross Sharpe collapsed" is a crutch phrase.** Correctly deployed, but deployed reflexively — the agent reaches for this whenever a config underperforms, sometimes before checking whether the cost model is the actual culprit.
- **Reaches for George-Hwang, Ang-Hodrick-Xing-Zhang, Frazzini-Pedersen, Bali-Cakici-Whitelaw by name.** Confers authority but can substitute for thinking about universe-specific deviations from those papers' original settings. Papers cited mostly apply to broader/smaller-cap universes than S&P 500.
- **Consistently predicts correlations too extreme (too positive or too negative in magnitude).** Run 4 predicted 0.85-0.95 |corr|, got 0.63. Run 7 self-reported correlation mean signed miss of –0.22. This is a documented bias now.
- **Tends to end reports with exactly 2-3 follow-up suggestions**, often including "regime-conditional version" and "orthogonalize against existing feature." Formulaic.
- **Mixes up decile convention at least once per multi-run session** ("decile 9 is typically the long side" — qualified with "typically" because it had to be verified empirically in Run 2).

## Single strongest example of quant-flavored thinking

From Run 3, catching `vol_adjusted_reversal`:

> "Superficially looks like a real, orthogonal alpha (correlation near zero with every existing feature) — which is *why due diligence matters*... 2020: IC mean 0.47, IC IR 55.7 — a correlation of 0.47 between feature and next-day returns is not a real-world effect; it's a numerical artifact. Most probable mechanism: during the March 2020 COVID chaos, the 21-day realized vol denominator collapsed for some names (trading halts, data gaps)..."

This is the full quant package: the agent resisted the surface appeal of orthogonality, went to year-by-year IC, recognized that 0.47 IC is physically impossible, and proposed a specific, testable mechanism tied to a known historical event. Nothing formulaic about it.

## Single weakest example

From Run 1, the initial pre-backtest green-light:

> "Given the orthogonality to existing features, these should genuinely add information. Let me run a combined backtest."

Two features correlated 0.34 with each other, both correlated –0.3 with `reversal_5d`, proposed at unit weight each on top of an existing 4-feature stack. The agent had all the information needed to predict the redundancy-with-wrong-sign outcome but didn't think through the aggregate sign budget. The subsequent post-mortem was excellent, but this moment — a pre-trade go decision based on pairwise-Pearson-vs-dominant-feature alone — is the recurring blind spot in its simplest form. A PM should read this as: *when this agent says "let me run the backtest now," it has often skipped the sign-budget check.*