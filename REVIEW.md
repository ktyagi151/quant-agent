# Per-run grades

**Run 1 — Feature proposal (overnight_drift, signed_vol)**
Goal: Propose 2 new price/volume features, pre-check, backtest, diagnose, keep/discard.
Grades: Ideation 4/5 | Diagnosis 4/5 | Iteration 4/5 | Self-correction 4/5 | Strategic 3/5 | Communication 4/5
Evidence: Strong economic motivation ("overnight returns carry continuation...intraday returns mean-revert") with cited literature. Nice post-hoc insight that "raw Pearson orthogonality does not imply portfolio complementarity" after spotting the -0.3 correlation with reversal_5d. Weakness: initial orthogonality check only against mom_12_1 was exactly the superficial screening the agent later critiques — the failure was foreseeable from the correlation matrix it had already printed.

**Run 2 — Knob tuning on baseline**
Goal: Find a weight/knob config beating +0.128 net Sharpe without new features.
Grades: Ideation 3/5 | Diagnosis 5/5 | Iteration 4/5 | Self-correction 5/5 | Strategic 4/5 | Communication 5/5
Evidence: Excellent mid-run mental-model revision on exit_n_deciles semantics: "I might have the direction backwards...higher exit values actually mean a *loose* threshold". The observation "IC IR stayed in the 1.08–1.18 band *even when gross Sharpe collapsed by 3×*" is genuinely sharp. Weakness: ideation was conventional (halflife, weight tilts) — never tried dropping mom_12_1 or exit=3 despite identifying both as obvious next steps.

**Run 3 — Due diligence on three registered features**
Goal: Vet three colleague features for legitimacy.
Grades: Ideation 4/5 | Diagnosis 5/5 | Iteration 4/5 | Self-correction 4/5 | Strategic 5/5 | Communication 5/5
Evidence: Textbook forensic work. Immediately pattern-matches "IC IR ≥ 2.6 every single year...peaking at 12.3" as "physics violation" / look-ahead signature; distinguishes it from the vol_adjusted_reversal failure which was a "**single-year artifact**, not global look-ahead." Proposes the exact likely bug (`pct_change(10)` without `.shift(1)`). Weakness: initial interpretation of vol_adjusted_reversal as "adding real orthogonal value" before running the per-year analysis shows the agent can be briefly fooled by aggregate metrics.

**Run 4 — Low-vol anomaly (BAB) test**
Goal: Test whether BAB is present / subsumed by baseline.
Grades: Ideation 3/5 | Diagnosis 5/5 | Iteration 3/5 | Self-correction 4/5 | Strategic 4/5 | Communication 5/5
Evidence: Clean pre-registration with numerical predictions, explicit keep-criteria set before the backtest. The U-shape diagnosis is sharp: "both tails outperform the middle, and the **high-vol decile actually earns more than the low-vol decile**" — correctly identifies this as IC-vs-tradable-alpha disconnect. Weakness: ideation was entirely scripted by the prompt (PM told it exactly what to operationalize), so this mostly exercises execution, not creativity.

**Run 5 — IC decay investigation**
Goal: Per-feature IC decay / regime diagnosis across 2014–2026.
Grades: Ideation 3/5 | Diagnosis 4/5 | Iteration 3/5 | Self-correction 3/5 | Strategic 5/5 | Communication 5/5
Evidence: Correctly follows protocol, produces actionable synthesis: "the strategy's net Sharpe ceiling is set by transaction costs on the fast signals, not by IC erosion." Nicely calls its own prior wrong ("I had specifically expected some [decay] in `mom_12_1`"). Weakness: small-sample stat claims ("reversal_5d 2025+2026 are weakest years") are treated almost equally with the 13-year conclusions — n=71 days for 2026 is noted but still influences the recommendation.

**Run 6 — Idiosyncratic vol (AHXZ) feature**
Goal: Test one non-trivial feature (idio vol) with proper look-ahead discipline.
Grades: Ideation 4/5 | Diagnosis 5/5 | Iteration 5/5 | Self-correction 4/5 | Strategic 5/5 | Communication 5/5
Evidence: Best iteration across runs — uses Run 4's low-vol failure as the explicit motivation ("Why should this work when `low_vol_252d` already failed?"). Clever vectorized implementation via `Var(resid) = Var(r_i)·(1 – Corr²)` avoiding per-stock loops. Names a new failure mode crisply: "a *smile* across deciles, not a *slope*...'content without tradability'." Weakness: only one combined-weight test (w=0.5) before discard; a w=0.25 check would have been cheap insurance.

---

# Synthesis

## Consistent strengths

- **Diagnosis of failure modes is this agent's standout skill.** Across all six runs it reliably distinguishes IC-vs-Sharpe disconnects, smile-vs-slope decile geometry, look-ahead bias signatures, regime artifacts, and cost-bound vs. signal-bound situations. It names these patterns explicitly rather than glossing ("physics violation," "content without tradability," "redundancy-with-wrong-sign failure").
- **Pre-registration discipline.** Runs 1, 4, and 6 explicitly state predicted correlations, sign expectations, and keep-criteria *before* backtesting, then honestly grade themselves against those predictions after ("-0.63 — lower than my -0.85 to -0.95 prediction"). This is exactly the practice junior researchers typically skip.
- **Communication of negative results.** Reports are structured, answer the PM's actual question, and deliver "discard" recommendations with as much rigor as positive ones. Follow-up suggestions are grounded in specific observed diagnostics, not generic fishing.
- **Tool-model self-correction.** Run 2's mid-flight reversal on `exit_n_deciles` semantics ("my mental model needs flipping") is the kind of update that most agents won't make.

## Consistent weaknesses

- **Pre-backtest screening is inconsistently thorough.** Run 1 checked orthogonality against mom_12_1 only, then ate a blow-up from the -0.3 correlation with reversal_5d that was *visible in the same correlation matrix it had just printed*. The agent later articulates the lesson beautifully — but the lesson does not prevent similar superficial screens in later runs (Run 4 does better because the prompt nudges it).
- **Small-sample discipline is loose.** Run 5 makes directional claims from n=71 days (2026 YTD) and a 3-year window without noting how much these could be noise. It flags the sample-size issue but still lets it influence the bottom line ("one more flat year and the case for keeping it weakens").
- **Iteration breadth is narrow within a run.** Run 2 tests 5 knob configs, all along axes of smoothing/weighting, never trying the obvious "drop mom_12_1" experiment it explicitly identifies as a follow-up. Once a hypothesis forms mid-run, the agent tends to close out rather than branch.
- **Occasional overreach in mechanism attribution.** "Low-vol was bid up as a bond proxy pre-2020 and crushed when rates spiked" (Run 6) is presented with confidence without any check against rates data. The story fits but isn't tested.

## What to trust the agent to do autonomously

- **Feature due diligence / red-team review** (Run 3 quality). Year-by-year IC audit, look-ahead detection, redundancy via correlation — ship this.
- **Clean ablation / IC-decay studies with pre-specified protocols** (Run 5). Follows the plan, produces a PM-ready table.
- **Writing up negative results.** Reports are well-structured, honest about limitations, and include actionable follow-ups. Minimal PM editing needed.
- **Post-hoc diagnostics on a surprising backtest result** — it reliably decomposes IC IR vs. gross vs. net vs. decile-monotonicity vs. regime.

## What to review / gate

- **Any "the feature passed pre-checks so I ran it" moment.** Have a human verify the pre-check covered every active feature, not just the expected-redundant one. This is the Run 1 failure mode and it will recur.
- **Feature proposals where the agent picks the direction itself** (Run 1 vs. scripted Runs 4/6). Ideation quality is fine but the screening for "which existing feature might this fight?" needs a checklist the agent does not reliably follow.
- **Claims that lean on 1–2 years of data.** Run 5's reversal_5d decay narrative is suggestive at best; don't let the agent act on it without a longer window or bootstrap.
- **Follow-up research proposals.** They're thoughtful but not costed — "decompose reversal into overnight/intraday" is one line in a report but a multi-day build. A PM should triage.

## Recurring blind spots / tics

- **Loves the IC-vs-Sharpe decoupling observation.** Uses it in Runs 1, 2, 4, 5, 6. It's a correct and important observation, but the agent reaches for it as a go-to frame even when a simpler explanation (e.g., "the decile spread just got noisier") would do.
- **Tends to conclude "the family is dead" after 1–2 failures.** Runs 4 and 6 jointly produce "**Don't re-attempt pure low-vol features in any form**" — a strong prior from n=2 attempts in one universe. Reasonable for this pipeline, but a PM should not let this ossify.
- **Predictions are anchored to textbook priors (AHXZ, FP, HFT decay hypothesis) and then "corrected" toward data.** When priors and data disagree it updates; but it rarely *starts* from a skeptical empirical prior.
- **Generous academic citations** (Frazzini-Pedersen, Ang-Hodrick-Xing-Zhang, Nagel, Bali-Cakici-Whitelaw, Kumar) — generally on-point, never obviously wrong, occasionally used as flourish rather than load-bearing argument.

## Single strongest example of quant-flavored thinking

From Run 3, diagnosing smoothed_momentum_v2:

> "IC IR ≥ 2.6 every single year from 2014 to 2026, mean ~5, peaking at 12.3 in 2014. Decile spread perfectly monotonic from −0.72 to +2.24 Sharpe. $1 → $31.36 net equity at only 25% daily turnover on a vanilla cross-sectional factor combo. Real alpha has year-to-year IC variance of maybe 50-100% of its mean with some near-zero years. This is a physics violation."

This combines the right quantitative benchmark (real IC IR magnitudes, variance-across-years as a lookahead tell), a confident pattern-match, and a specific hypothesized bug mechanism (`pct_change(10)` without `.shift(1)`). Exactly what a PM wants from a due-diligence pass.

## Single weakest example

From Run 1, the pre-backtest green-light:

> "Orthogonal to `mom_12_1` (0.069 and 0.008 — *much* more orthogonal than I expected, which is a positive surprise; they aren't just repackaged momentum)... Given the orthogonality to existing features, these should genuinely add information. Let me run a combined backtest."

The correlation matrix the agent had just computed showed **-0.28 and -0.35 against reversal_5d**, right next to the mom_12_1 numbers it chose to celebrate. The agent ignored the active feature it was about to fight, ran an unnecessarily expensive backtest, and only diagnosed the issue post-hoc. It then wrote an excellent lesson about exactly this mistake — but the lesson is post-mortem, not prophylactic. A human reviewer reading only the pre-backtest block could have saved the run.