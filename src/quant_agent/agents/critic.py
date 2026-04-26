"""Critic agent — final gate before a candidate is recorded as a real result.

The critic has VETO power. It reviews:
  * the alpha agent's reasoning (was the methodology sound?)
  * the portfolio agent's choice (was the construction method appropriate?)
  * the cost/risk agent's metrics (any hard-constraint borderline cases?)
  * governance flags from the IPS (e.g. inflated Sharpe → look-ahead audit)

Output: APPROVED, AUDIT_REQUIRED, or VETOED. AUDIT_REQUIRED triggers a manual
review queue; VETOED kills the candidate. Only APPROVED candidates are
forwarded to the meta-agent for OOS evaluation.

This is an extension of the existing review.py (which grades whole sessions);
the critic operates on a single candidate within a cycle.
"""
from __future__ import annotations

import json

from anthropic import beta_tool

from ..ips import IPS, to_yaml_summary
from .base import AgentSpec


def _system_prompt(ips: IPS, **_ctx) -> str:
    return f"""You are the **critic agent** — the final gate before a \
candidate is forwarded to the meta-agent for out-of-sample evaluation.

# Investment Policy Statement
{to_yaml_summary(ips)}

# Your job
You will receive:
  - the alpha agent's signal + reasoning + diagnosis
  - the portfolio agent's chosen construction method + parameters
  - the cost/risk agent's metrics report + verdict

Produce one of three verdicts:

**APPROVED** — methodology was sound, IPS constraints satisfied, no governance flags. Forward to meta-agent.

**AUDIT_REQUIRED** — methodology questionable or governance flag triggered. Specific triggers:
  - In-sample standalone Sharpe > {ips.governance.flag_inflated_sharpe_threshold}: possible look-ahead, request manual code audit
  - In-sample IC IR > {ips.governance.flag_inflated_ic_ir_threshold}: possible look-ahead, request year-by-year IC check
  - The alpha agent reused a feature already known to fail: requires explanation
  - Pre-registered prediction was missed by >2× the predicted range: calibration concern
The candidate is NOT advanced; it goes to a queue for human review.

**VETOED** — kill the candidate. Triggers:
  - Hard IPS constraint violation that the cost/risk agent missed
  - Look-ahead bug detected in feature code
  - Methodology fundamentally unsound (e.g. used reversal_5d at zero weight while still claiming reversal alpha)
  - Cost model misapplied (e.g. flat 5bps used when IPS specifies sqrt-impact)

# Calibration discipline
You also have access to the calibration_report. If the alpha agent's recent \
predictions show systematic over-optimism, weight new claims accordingly: \
ask for tighter evidence on net_sharpe claims if the agent's net_sharpe \
hit rate is below 50%.

# What you DO NOT do
- You do not propose changes to features or weights — only verdict + reasoning.
- You do not access the holdout — only the meta-agent does, after your approval.

# Output (JSON)
{{
  "verdict": "APPROVED" | "AUDIT_REQUIRED" | "VETOED",
  "confidence": 0.0-1.0,
  "reasons": ["..."],
  "audit_items": ["..."],   // if AUDIT_REQUIRED
  "veto_specifics": "..."    // if VETOED
}}"""


def _tool_factory(session=None, ips: IPS | None = None, **_ctx) -> list:
    if session is None or ips is None:
        return []

    @beta_tool
    def review_candidate(candidate_summary: str) -> str:
        """Run governance flags on the candidate's metrics summary.

        candidate_summary: JSON string with at least {sharpe, gross_sharpe,
        ic_ir, hard_violations, soft_scores}.
        Returns a JSON dict of governance flags raised. The agent then
        synthesizes the final verdict.
        """
        try:
            d = json.loads(candidate_summary)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid candidate_summary JSON: {e}"})
        flags = []
        if d.get("sharpe", 0) > ips.governance.flag_inflated_sharpe_threshold:
            flags.append(f"sharpe={d['sharpe']:.3f} exceeds inflation threshold "
                        f"{ips.governance.flag_inflated_sharpe_threshold} — possible look-ahead")
        if d.get("ic_ir", 0) > ips.governance.flag_inflated_ic_ir_threshold:
            flags.append(f"ic_ir={d['ic_ir']:.3f} exceeds inflation threshold "
                        f"{ips.governance.flag_inflated_ic_ir_threshold} — possible look-ahead")
        if d.get("hard_violations"):
            flags.append(f"{len(d['hard_violations'])} hard IPS violations present — VETO")
        if d.get("gross_sharpe", 0) > 0 and d.get("sharpe", 0) < 0:
            flags.append("gross Sharpe positive but net Sharpe negative — cost-bound, not signal-bound")
        return json.dumps({"governance_flags": flags, "n_flags": len(flags)}, default=str)

    @beta_tool
    def calibration_check() -> str:
        """Pull the calibration track record. The critic uses this to weight
        the alpha agent's predictions: if recent net_sharpe predictions miss
        consistently, the critic should request stronger evidence."""
        if session.calibration is None:
            return json.dumps({"error": "no calibration store"})
        return json.dumps(session.calibration.summary(), indent=2, default=str)

    return [review_candidate, calibration_check]


critic_agent_spec = AgentSpec(
    name="critic",
    description="Final gate before meta-agent — APPROVED / AUDIT_REQUIRED / VETOED",
    system_prompt_fn=_system_prompt,
    tool_factory=_tool_factory,
    model="claude-opus-4-7",
    use_thinking=True,
    effort="high",     # critical decision; prioritize correctness
    max_tokens=4000,
    max_iterations=4,
)
