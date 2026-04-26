"""Cost & risk agent — applies the IPS cost model, computes all risk
metrics, and reports any hard-constraint violations.

This agent is mostly mechanical (the heavy lifting is in cost_models.py and
risk.py). Its LLM job is to interpret the metrics and either approve the
candidate portfolio or hand it back to the portfolio agent with a specific
fix request.
"""
from __future__ import annotations

import json

from anthropic import beta_tool

from ..ips import IPS, to_yaml_summary
from .base import AgentSpec


def _system_prompt(ips: IPS, **_ctx) -> str:
    return f"""You are the **cost & risk agent**. You receive a candidate \
portfolio (weights matrix) from the portfolio agent and produce a verdict: \
APPROVED, REPAIRABLE, or REJECTED.

# Investment Policy Statement (binding)
{to_yaml_summary(ips)}

# Your job
1. Compute the IPS cost model against the candidate weights. Produce per-day \
   net returns after the realistic cost model (e.g. sqrt-impact), not the \
   pessimistic flat model.
2. Compute every metric the IPS hard constraints reference: gross, net, \
   sector exposures, single-name caps, breadth, turnover, drawdown.
3. Compare metrics to constraints. Any HARD constraint violation = REJECTED.
4. If REJECTED, name the specific violation and the closest feasible repair: \
   e.g. 'sector cap on Tech violated at 0.34, reduce by trimming top 3 long \
   names by ~15% each'. Hand back to portfolio agent.
5. If approved, hand to critic agent with metrics report attached.

# Output structure (JSON)
{{
  "verdict": "APPROVED" | "REPAIRABLE" | "REJECTED",
  "metrics": {{ ... }},
  "hard_violations": [...],
  "soft_scores": {{ ... }},
  "recommended_repair": "...",   // only if REPAIRABLE
  "notes": "..."                  // anything the critic should know
}}

# Hard rule
You may NEVER pass a portfolio with a hard violation as APPROVED. The IPS is \
non-negotiable. If REPAIRABLE, the portfolio agent gets one revision; if it \
fails twice, escalate to REJECTED."""


def _tool_factory(session=None, ips: IPS | None = None, **_ctx) -> list:
    if session is None or ips is None:
        return []

    @beta_tool
    def evaluate_candidate_portfolio() -> str:
        """Compute realistic cost-adjusted returns + IPS metrics + violations."""
        last = session.last_run() if hasattr(session, "last_run") else None
        if last is None:
            return json.dumps({"error": "no candidate portfolio to evaluate"})

        from ..cost_models import cost_model_from_ips
        from ..risk import risk_report

        weights = last.weights
        cm = cost_model_from_ips(ips.cost_model)
        prior = weights.shift(1).fillna(0.0)
        per_day_cost = cm.cost_per_day(
            weights=weights,
            prior_weights=prior,
            dollar_vol=session.feature_cache.get("dollar_vol_20d"),
        )
        gross_returns = last.gross_returns
        net_after_realistic = gross_returns - per_day_cost.reindex_like(gross_returns).fillna(0)

        report = risk_report(weights, net_after_realistic, session.sectors, ips)
        report["realistic_net_sharpe"] = float(
            net_after_realistic.mean() / max(net_after_realistic.std(ddof=0), 1e-9) * (252 ** 0.5)
        )
        report["realistic_avg_cost_drag_ann"] = float(per_day_cost.mean() * 252)
        verdict = "APPROVED" if report["passes_ips"] else "REJECTED"
        report["verdict"] = verdict
        return json.dumps(report, indent=2, default=str)

    return [evaluate_candidate_portfolio]


cost_risk_agent_spec = AgentSpec(
    name="cost_risk",
    description="Applies realistic cost model + IPS hard-constraint enforcement",
    system_prompt_fn=_system_prompt,
    tool_factory=_tool_factory,
    model="claude-opus-4-7",
    use_thinking=True,
    effort="medium",   # mostly mechanical; doesn't need max effort
    max_tokens=4000,
    max_iterations=4,
)
