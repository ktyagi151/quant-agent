"""Meta-agent — the compounding loop. Promotes prompt rewrites and
features only when they survive an IPS-protected holdout test.

Operates on a longer cadence than the per-cycle agents (e.g. weekly).
Reads:
  - the journal (every prior run)
  - the calibration history (forecast errors)
  - the prompt history (every prior version of each agent's system prompt)

Proposes:
  - additions / removals / modifications to agent system prompts
  - feature graveyard (features that have failed N times in a row)

Decides via the holdout: a proposed prompt rewrite is promoted iff:
  * it improves out-of-sample Sharpe on the holdout window
  * the improvement survives Bonferroni correction over recent attempts
  * no new IPS hard constraint becomes binding
"""
from __future__ import annotations

import json

from anthropic import beta_tool

from ..ips import IPS, to_yaml_summary
from .base import AgentSpec


def _system_prompt(ips: IPS, **_ctx) -> str:
    n_tests = ips.holdout.bonferroni_n_tests
    return f"""You are the **meta-agent**. You evolve the multi-agent system \
itself: you rewrite the alpha / portfolio / cost-risk / critic agents' \
system prompts based on accumulated evidence, and you maintain a feature \
graveyard so the system stops re-attempting known dead-ends.

# Investment Policy Statement (binding)
{to_yaml_summary(ips)}

# Your evidence sources
- journal: every backtest run, with config + metrics + which agent led the work
- calibration history: each agent's per-type prediction hit rate + bias
- prompt_history: every prior version of each agent's system prompt with \
  the period it was active and the metrics achieved during that period
- the IPS holdout window — frozen at \
  {ips.holdout.holdout_start} to {ips.holdout.holdout_end}, ≥{ips.holdout.min_observations} obs.

# Your decision rule (HARD)
A prompt rewrite or feature promotion may only happen if BOTH:
  (a) Out-of-sample improvement on the holdout exceeds Bonferroni-corrected \
      significance with n={n_tests}: a rough rule of thumb is the Sharpe \
      difference must exceed ~{2.0 / (n_tests ** 0.5):.2f} × the in-sample noise std \
      to be considered significant after correction.
  (b) No new hard IPS constraint becomes binding under the proposed change.

# What you do
1. Review the journal: which agents have been performing best vs their goals?
2. Review calibration: which agents have systematic biases that could be \
   addressed with prompt edits?
3. Propose ONE concrete prompt rewrite to ONE agent. Articulate the precise \
   change and the hypothesized improvement.
4. Test the rewrite: re-run the relevant past goals with the proposed prompt \
   on the IN-SAMPLE window only. If improvements are seen, request the \
   orchestrator run an OOS validation on the holdout.
5. Promote ONLY IF the OOS test passes per the decision rule above.

# What you DO NOT do
- You do not see the holdout outputs directly until the orchestrator's \
  validation step grants you read-only access to the metrics.
- You do not propose new features — that's the alpha agent's role.
- You do not change the IPS — only humans do that.

# Failure-to-promote is the default
The vast majority of rewrites should NOT promote. A 5-10% promotion rate is \
healthy — anything higher is a sign you're overfitting the holdout."""


def _tool_factory(session=None, ips: IPS | None = None, **_ctx) -> list:
    if session is None or ips is None:
        return []

    @beta_tool
    def list_journal_runs(limit: int = 20) -> str:
        """Return the most recent N runs from the journal."""
        if session.journal is None:
            return json.dumps({"error": "no journal"})
        runs = session.journal.all_runs()
        recent = runs[-limit:]
        return json.dumps([
            {
                "run_id": r.get("run_id"),
                "timestamp": r.get("timestamp"),
                "config": r.get("config"),
                "summary": {
                    k: r.get("summary", {}).get(k)
                    for k in ["sharpe", "gross_sharpe", "ic_ir", "avg_turnover", "max_drawdown"]
                },
            }
            for r in recent
        ], indent=2, default=str)

    @beta_tool
    def calibration_summary() -> str:
        """Return the cross-session calibration summary."""
        if session.calibration is None:
            return json.dumps({"error": "no calibration store"})
        return json.dumps(session.calibration.summary(), indent=2, default=str)

    @beta_tool
    def list_prompt_versions(agent_name: str) -> str:
        """List all prompt versions for the named agent (alpha, portfolio, cost_risk, critic, meta).

        Returns version numbers, promoted/rolled-back status, and metrics
        recorded at promotion time. Use this to understand the prompt history
        of an agent before proposing a rewrite.
        """
        from ..prompt_history import PromptHistory

        store = PromptHistory.default()
        versions = store.list_versions(agent_name)
        return json.dumps([
            {
                "version": v.version,
                "parent_version": v.parent_version,
                "rationale": v.rationale,
                "created_at": v.created_at,
                "promoted": v.promoted,
                "rolled_back": v.rolled_back,
                "metrics_at_promotion": v.metrics_at_promotion,
            }
            for v in versions
        ], indent=2, default=str)

    @beta_tool
    def propose_prompt_rewrite(agent_name: str, new_text: str, rationale: str) -> str:
        """Submit a candidate system-prompt rewrite for an agent.

        The proposal is stored in the prompt-history store as a pending
        version with a parent pointer to the current latest. It is NOT
        promoted automatically — a separate validation step runs the
        proposed prompt against the holdout window and applies the
        Bonferroni-corrected promotion rule before any change takes effect.

        Args:
            agent_name: One of alpha, portfolio, cost_risk, critic, meta.
            new_text: The full new system prompt text. Must be a complete
                replacement, not a diff.
            rationale: One-paragraph justification — what specific behavior
                does this rewrite intend to change, and what evidence from
                the journal/calibration history motivated it?

        Returns the assigned version number + a status string.
        """
        from ..prompt_history import PromptHistory

        valid_agents = {"alpha", "portfolio", "cost_risk", "critic", "meta"}
        if agent_name not in valid_agents:
            return json.dumps({
                "error": f"unknown agent_name {agent_name!r}; must be one of {sorted(valid_agents)}",
            })
        if not new_text or len(new_text) < 200:
            return json.dumps({
                "error": "new_text must be a complete prompt (≥200 chars), not a diff or fragment",
            })
        if not rationale or len(rationale) < 30:
            return json.dumps({
                "error": "rationale must be a substantive justification (≥30 chars)",
            })

        store = PromptHistory.default()
        parent = store.latest(agent_name)
        pv = store.add_proposal(
            agent=agent_name,
            text=new_text,
            rationale=rationale,
            parent=parent,
        )
        return json.dumps({
            "status": "PROPOSED_PENDING_VALIDATION",
            "agent": agent_name,
            "version": pv.version,
            "parent_version": pv.parent_version,
            "text_hash": pv.text_hash,
            "next_step": "Orchestrator will run validation on holdout. Promotion is gated on Bonferroni-corrected OOS improvement.",
        }, indent=2, default=str)

    return [
        list_journal_runs,
        calibration_summary,
        list_prompt_versions,
        propose_prompt_rewrite,
    ]


meta_agent_spec = AgentSpec(
    name="meta",
    description="Evolves agent prompts and the feature graveyard via OOS-protected promotion",
    system_prompt_fn=_system_prompt,
    tool_factory=_tool_factory,
    model="claude-opus-4-7",
    use_thinking=True,
    effort="high",
    max_tokens=12000,
    max_iterations=6,
)
