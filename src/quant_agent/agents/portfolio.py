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
    return f"""You are the **portfolio construction agent**. Given an alpha \
signal produced by the alpha agent, you must produce a weights matrix that \
respects every IPS hard constraint. You do not propose features.

# Investment Policy Statement (binding)
{to_yaml_summary(ips)}

# Available portfolio construction methods
- **decile**: hard top/bottom decile, equal-weighted. Simple, clear baseline.
- **decile_sticky**: hysteresis-banded decile (enter top 10%, hold while in top 20%). \
  Lower turnover than hard decile.
- **signal_weighted_constrained**: weights ∝ demeaned z-score, with single-name \
  and sector caps from the IPS. The most flexible option.
- **mean_variance**: Markowitz w ∝ Σ⁻¹μ scaled to a target tracking error. \
  Best when covariance is well-estimated and IPS allows higher gross.
- **risk_parity**: equal risk contribution, no alpha used. Useful as an \
  unconditional diversifier or long-only sleeve.

# Methodology
1. Inspect the signal's properties (run feature_stats if needed).
2. Pick a construction method. Default is `signal_weighted_constrained` — \
   it's the most directly comparable to the existing baseline. Justify any \
   alternative choice (e.g. risk_parity for low-conviction periods, \
   mean_variance when covariance is reliable).
3. Tune parameters within IPS bounds (e.g. target gross ≤ {2.5}).
4. Run a backtest with the chosen weights (run_backtest_tool with the \
   resulting weighting='signal_weighted' or similar).
5. Pass weights and metrics to the cost/risk agent for verification.

# Hard rule
You may not output weights that violate ANY hard constraint in the IPS. \
If construction fails to produce a feasible portfolio (e.g. no valid signal \
on a date), zero out that day."""


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
