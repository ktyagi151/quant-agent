"""Portfolio construction agent — given an alpha signal + IPS, choose
a portfolio construction method (decile L/S, signal-weighted-constrained,
mean-variance, risk-parity) and produce IPS-respecting weights.
"""
from __future__ import annotations

import json

from anthropic import beta_tool

from ..ips import IPS, to_yaml_summary
from .base import AgentSpec


def _system_prompt(ips: IPS, **_ctx) -> str:
    # Detect dollar-neutral mandate so we can warn about long-only methods.
    net_neutral = any(
        hc.metric == "net_exposure" and hc.op in ("<=", "<") and hc.threshold <= 0.15
        for hc in ips.hard_constraints
    )
    net_cap = next(
        (hc.threshold for hc in ips.hard_constraints if hc.metric == "net_exposure"),
        None,
    )
    long_only_warning = (
        f"\n  - **Critical**: this is LONG-ONLY by construction. Net exposure = gross exposure = 1.0. "
        f"It WILL violate the IPS net_exposure cap (≤ {net_cap}) unless explicitly paired with a "
        f"short overlay you construct yourself. **Do not use as a fallback** when the alpha signal "
        f"failed — it produces a different mandate, not a degraded version."
    ) if net_neutral else ""

    return f"""You are the **portfolio construction agent**. Given an alpha signal produced by the alpha agent, you must produce a weights matrix that respects every IPS hard constraint. You do not propose features.

# Investment Policy Statement (binding)
{to_yaml_summary(ips)}

# Available portfolio construction methods — strengths and the trap each carries

- **decile**: hard top/bottom decile, equal-weighted. Dollar-neutral by construction. Simple baseline.
- **decile_sticky**: hysteresis-banded decile (enter top 10%, hold while in top 20%). Dollar-neutral. Lower turnover than hard decile.
- **signal_weighted_constrained**: weights ∝ demeaned z-score, with single-name and sector caps from the IPS. **Dollar-neutral by construction (mean-zero) AND respects all IPS caps natively.** This is the safest choice for a net-neutral mandate and should be your default.
- **mean_variance**: Markowitz w ∝ Σ⁻¹μ scaled to a target tracking error.
  - **Trap**: on small universes (≤ 50 names) or when covariance is poorly estimated, the optimizer can produce degenerate solutions concentrated in 1–2 names with extreme weights. Always check max_abs_weight on the result before passing downstream — if any name has |w| > 0.20 with a 0.05 IPS cap, the construction has failed and the result must NOT be used.
  - **Trap**: target_te is annualized — if covariance is sparse or recent history is anomalous, this scales unpredictably.
  - Use when: large universe, well-conditioned covariance, IPS allows higher gross.
- **risk_parity**: equal risk contribution, **no alpha used, LONG-ONLY by construction**.{long_only_warning}
  - Use ONLY for: long-only sleeves, benchmark comparison, or paired with an explicit short overlay.

# Methodology

1. **Read the alpha agent's report carefully.** What is the signal? Did the backtest succeed (positive IC IR, monotonic deciles) or fail?
2. **If alpha succeeded**: use `signal_weighted_constrained` as your default. Tune `target_gross` within IPS gross cap; rely on IPS-derived `max_single_name` and `max_sector_weight`. Justify any deviation in writing.
3. **If alpha failed**: do NOT fabricate a portfolio. Two valid actions:
   - Construct from the most recent successful signal in the registry (the journal can tell you which one). Do not invent weights from a failed signal.
   - Or escalate "no compliant portfolio available" — the critic will route to AUDIT_REQUIRED, which is the correct outcome. **Do NOT switch to risk_parity to "do something."** Under a dollar-neutral mandate it is not a valid fallback.
4. **Verify the candidate**: check `max_abs_weight` on the result. If degenerate (one or two names dominate), the optimizer failed — try a different method or report inability to construct.
5. **Always store** your output via `session.candidate_weights` (the construct_* tools do this for you). The cost/risk agent reads from there.

# Hard rule

You may not output weights that violate ANY hard IPS constraint. If you cannot produce a compliant portfolio:
- A long-only construction (risk_parity) is **not** a valid fallback under a dollar-neutral IPS.
- A degenerate mean-variance solution (one name >>5% weight) is **not** a valid output — try signal_weighted_constrained instead.
- "I could not construct a compliant portfolio because [reason]" is a valid output, and is preferable to anything that violates the IPS.

When in doubt: signal_weighted_constrained on the strongest available signal in the registry."""


def _tool_factory(session=None, **_ctx) -> list:
    if session is None:
        return []

    @beta_tool
    def construct_signal_weighted(target_gross: float = 2.0, max_single_name: float = 0.05,
                                  max_sector_weight: float = 0.30) -> str:
        """Construct signal-weighted-constrained weights from the most recent signal.

        Persists the result to session.candidate_weights so cost_risk agent
        can evaluate THIS portfolio (not alpha's backtest weights).
        """
        last = session.last_run() if hasattr(session, "last_run") else None
        if last is None:
            return json.dumps({"error": "no signal available; alpha agent must run first"})
        from ..optimization import signal_weighted_constrained

        weights = signal_weighted_constrained(
            signal=last.signal,
            sectors=session.sectors,
            target_gross=target_gross,
            max_single_name=max_single_name,
            max_sector_weight=max_sector_weight,
        )
        session.candidate_weights = weights
        session.candidate_weights_method = "signal_weighted_constrained"
        return json.dumps({
            "method": "signal_weighted_constrained",
            "shape": list(weights.shape),
            "avg_gross": float(weights.abs().sum(axis=1).mean()),
            "avg_n_long": float((weights > 0).sum(axis=1).mean()),
            "avg_n_short": float((weights < 0).sum(axis=1).mean()),
            "max_abs_weight": float(weights.abs().max().max()),
            "stored": "session.candidate_weights",
        }, default=str)

    @beta_tool
    def construct_mean_variance(target_te: float = 0.10, lookback: int = 252) -> str:
        """Construct mean-variance weights with target tracking-error.

        Persists the result to session.candidate_weights.
        """
        last = session.last_run() if hasattr(session, "last_run") else None
        if last is None:
            return json.dumps({"error": "no signal available; alpha agent must run first"})
        prices = session.panel["adj_close"]
        rets = prices.pct_change()
        from ..optimization import mean_variance_weights

        w = mean_variance_weights(last.signal, rets, target_te=target_te, lookback=lookback)
        session.candidate_weights = w
        session.candidate_weights_method = "mean_variance"
        return json.dumps({
            "method": "mean_variance",
            "shape": list(w.shape),
            "avg_gross": float(w.abs().sum(axis=1).mean()),
            "avg_n_long": float((w > 0).sum(axis=1).mean()),
            "avg_n_short": float((w < 0).sum(axis=1).mean()),
            "max_abs_weight": float(w.abs().max().max()),
            "stored": "session.candidate_weights",
        }, default=str)

    @beta_tool
    def construct_risk_parity(lookback: int = 252) -> str:
        """Construct risk-parity weights (no signal needed).

        Persists the result to session.candidate_weights.
        """
        prices = session.panel["adj_close"]
        rets = prices.pct_change()
        from ..optimization import risk_parity_weights

        w = risk_parity_weights(rets, lookback=lookback)
        session.candidate_weights = w
        session.candidate_weights_method = "risk_parity"
        return json.dumps({
            "method": "risk_parity",
            "shape": list(w.shape),
            "avg_gross": float(w.abs().sum(axis=1).mean()),
            "avg_n_long": float((w > 0).sum(axis=1).mean()),
            "stored": "session.candidate_weights",
        }, default=str)

    return [construct_signal_weighted, construct_mean_variance, construct_risk_parity]


portfolio_agent_spec = AgentSpec(
    name="portfolio",
    description="Constructs IPS-compliant weights from an alpha signal",
    system_prompt_fn=_system_prompt,
    tool_factory=_tool_factory,
    model="claude-opus-4-7",
    use_thinking=True,
    effort="high",
    max_tokens=8000,
    max_iterations=8,
)
