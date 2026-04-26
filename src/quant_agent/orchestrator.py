"""Multi-agent pipeline coordinator.

Runs a single "investment cycle" through:

  alpha → portfolio → cost_risk → critic → (commit or veto)

Then optionally invokes the meta-agent on a slower cadence (weekly).

This module wires AgentSpec objects to the Anthropic API (for live runs) AND
provides a fully-stubbed dry-run mode that exercises the orchestration logic
without API calls — useful for testing the pipeline shape and for offline
inspection of system prompts / IPS rendering.

The orchestrator owns:
  - the IPS instance
  - the ResearchSession (loaded once per cycle, slicing data to in-sample only
    for alpha / portfolio / cost_risk)
  - the prompt history store
  - dispatching each agent and routing outputs

Each agent is a self-contained API call (or stub). The orchestrator does NOT
share Anthropic conversation context between agents; outputs from one agent
are passed as structured data into the next agent's user message.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from . import agents as ag
from .agents.base import AgentResult, AgentSpec
from .ips import IPS, load_ips, validate_ips
from .io_utils import outputs_dir
from .prompt_history import PromptHistory


# ----- runtime configuration -----------------------------------------------


@dataclass
class CycleResult:
    cycle_id: str
    started_at: str
    finished_at: str
    ips_name: str
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    final_verdict: str = "pending"   # APPROVED | AUDIT_REQUIRED | VETOED | ERROR
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "ips_name": self.ips_name,
            "agent_results": {
                name: {
                    "agent_name": r.agent_name,
                    "success": r.success,
                    "outputs": r.outputs,
                    "usage": r.usage,
                    "error": r.error,
                }
                for name, r in self.agent_results.items()
            },
            "final_verdict": self.final_verdict,
            "notes": self.notes,
        }


# ----- session loading ------------------------------------------------------


def load_session_for_ips(ips: IPS, limit: int | None = None):
    """Build a ResearchSession sliced to the IN-SAMPLE window only.

    The orchestrator never lets agents see the holdout. The meta-agent is the
    only role that gets holdout access, and it does so via a separate session
    loaded with the holdout window.
    """
    from .agent_tools import ResearchSession

    return ResearchSession.from_cache(
        start=ips.holdout.in_sample_start,
        end=ips.holdout.in_sample_end,
        limit=limit,
        liquidity_threshold=ips.universe.min_dollar_vol_usd,
        cost_bps=ips.cost_model.flat_bps,    # backtest-internal default; cost_risk applies the real model
        n_deciles=10,
        journal=True,
    )


def load_holdout_session(ips: IPS, limit: int | None = None):
    """Separate session sliced to the holdout window. Used ONLY by meta-agent."""
    from .agent_tools import ResearchSession

    return ResearchSession.from_cache(
        start=ips.holdout.holdout_start,
        end=ips.holdout.holdout_end,
        limit=limit,
        liquidity_threshold=ips.universe.min_dollar_vol_usd,
        cost_bps=ips.cost_model.flat_bps,
        n_deciles=10,
        journal=True,    # uses same journal — meta-agent reads but orchestrator ensures no in-sample leakage
    )


# ----- live agent invocation -----------------------------------------------


def _invoke_agent_live(
    spec: AgentSpec,
    ips: IPS,
    user_message: str,
    session=None,
) -> AgentResult:
    """Execute a single agent against the live Anthropic API.

    Returns a uniform AgentResult so downstream code is dispatch-agnostic.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return AgentResult(
            agent_name=spec.name,
            success=False,
            outputs={},
            error="ANTHROPIC_API_KEY not set",
        )
    from anthropic import Anthropic

    client = Anthropic()
    tools = spec.build_tools(session=session, ips=ips)
    is_haiku = "haiku" in spec.model.lower()
    extra: dict = {}
    if not is_haiku and spec.use_thinking:
        extra["thinking"] = {"type": "adaptive", "display": "summarized"}
        extra["output_config"] = {"effort": spec.effort}

    runner = client.beta.messages.tool_runner(
        model=spec.model,
        max_tokens=spec.max_tokens,
        max_iterations=spec.max_iterations,
        system=[
            {
                "type": "text",
                "text": spec.render_system_prompt(ips, session=session),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=tools or None,
        messages=[{"role": "user", "content": user_message}],
        **extra,
    )

    transcript: list[dict] = []
    final_text_parts: list[str] = []
    total_input = total_output = total_cache_read = 0
    for msg in runner:
        u = msg.usage
        total_input += u.input_tokens
        total_output += u.output_tokens
        total_cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        record = {"stop_reason": msg.stop_reason, "content": []}
        for b in msg.content:
            btype = getattr(b, "type", None)
            if btype == "text":
                record["content"].append({"type": "text", "text": b.text})
                final_text_parts.append(b.text)
            elif btype == "thinking":
                record["content"].append({"type": "thinking", "text": getattr(b, "thinking", "")})
            elif btype == "tool_use":
                record["content"].append({"type": "tool_use", "name": b.name, "input": b.input})
        transcript.append(record)

    return AgentResult(
        agent_name=spec.name,
        success=True,
        outputs={"final_text": "\n".join(final_text_parts)},
        transcript=transcript,
        usage={
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_read_input_tokens": total_cache_read,
        },
    )


# ----- dry-run agent invocation --------------------------------------------


def _invoke_agent_dry_run(spec: AgentSpec, ips: IPS, user_message: str, **_ctx) -> AgentResult:
    """Stub: render the prompt + tool list but make no API call.

    Used by orchestrate --dry-run. Lets us inspect prompt rendering, IPS
    summarization, and tool wiring without spending budget.
    """
    rendered = spec.render_system_prompt(ips)
    return AgentResult(
        agent_name=spec.name,
        success=True,
        outputs={
            "system_prompt_chars": len(rendered),
            "system_prompt_preview": rendered[:400] + ("..." if len(rendered) > 400 else ""),
            "user_message_preview": user_message[:300] + ("..." if len(user_message) > 300 else ""),
            "would_invoke": f"client.beta.messages.tool_runner(model='{spec.model}', ...)",
        },
        transcript=[],
        usage={"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
    )


# ----- the cycle ------------------------------------------------------------


def run_cycle(
    ips: IPS,
    session=None,
    dry_run: bool = True,
    invoker: Callable | None = None,
) -> CycleResult:
    """Run one full alpha → portfolio → cost_risk → critic cycle.

    `dry_run=True` (default) skips API calls and only renders prompts.
    `invoker` optionally overrides the agent invocation path (for testing).
    """
    cycle_id = datetime.utcnow().strftime("cycle_%Y%m%d_%H%M%S")
    started_at = datetime.utcnow().isoformat(timespec="seconds")

    if invoker is None:
        invoker = _invoke_agent_dry_run if dry_run else _invoke_agent_live

    result = CycleResult(
        cycle_id=cycle_id,
        started_at=started_at,
        finished_at="",
        ips_name=ips.name,
    )

    pipeline = [
        (ag.alpha_agent_spec, "Propose and test ONE feature on the in-sample window. Follow the IPS methodology."),
        (ag.portfolio_agent_spec, "Construct an IPS-compliant portfolio from the alpha agent's signal."),
        (ag.cost_risk_agent_spec, "Evaluate the candidate portfolio under the IPS cost model + hard constraints."),
        (ag.critic_agent_spec, "Final verdict: APPROVED, AUDIT_REQUIRED, or VETOED."),
    ]

    # Thread prior agent outputs forward so each agent sees what the previous
    # agents produced. Capped per-agent to keep input tokens bounded.
    PER_AGENT_PRIOR_CHARS = 2500
    prior_outputs: dict[str, str] = {}

    for spec, default_msg in pipeline:
        user_msg = default_msg
        if prior_outputs:
            sections = ["", "# Prior agent outputs in this cycle (read-only context)"]
            for prev_name, prev_text in prior_outputs.items():
                sections.append(f"\n## {prev_name} agent said:")
                sections.append(prev_text[:PER_AGENT_PRIOR_CHARS] +
                                ("\n... [truncated]" if len(prev_text) > PER_AGENT_PRIOR_CHARS else ""))
            user_msg += "\n".join(sections)

        try:
            r = invoker(spec=spec, ips=ips, user_message=user_msg, session=session)
        except Exception as e:  # noqa: BLE001
            r = AgentResult(agent_name=spec.name, success=False, outputs={}, error=f"{type(e).__name__}: {e}")
        result.agent_results[spec.name] = r
        # Capture text for the next agent's context.
        prior_outputs[spec.name] = r.outputs.get("final_text", "") or ""
        if not r.success:
            result.final_verdict = "ERROR"
            result.notes = f"{spec.name} failed: {r.error}"
            break

    if result.final_verdict == "pending":
        # In live mode, the critic's final_text drives this. In dry-run we just say PENDING.
        critic_out = result.agent_results.get("critic")
        if critic_out and "final_text" in critic_out.outputs:
            text = critic_out.outputs["final_text"].upper()
            if "VETOED" in text:
                result.final_verdict = "VETOED"
            elif "AUDIT_REQUIRED" in text:
                result.final_verdict = "AUDIT_REQUIRED"
            elif "APPROVED" in text:
                result.final_verdict = "APPROVED"
            else:
                result.final_verdict = "AUDIT_REQUIRED"
        else:
            result.final_verdict = "DRY_RUN"

    result.finished_at = datetime.utcnow().isoformat(timespec="seconds")
    return result


# ----- meta-agent cadence ---------------------------------------------------


def run_meta_pass(ips: IPS, session=None, dry_run: bool = True) -> AgentResult:
    """Invoke the meta-agent. Run on a longer cadence (e.g. weekly)."""
    invoker = _invoke_agent_dry_run if dry_run else _invoke_agent_live
    return invoker(
        spec=ag.meta_agent_spec,
        ips=ips,
        user_message=(
            "Review the journal + calibration history. Propose ONE prompt rewrite "
            "or feature graveyard addition. Do not promote without an OOS test on the holdout."
        ),
        session=session,
    )


# ----- save -----------------------------------------------------------------


def save_cycle(cycle: CycleResult, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = outputs_dir() / cycle.cycle_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cycle.json").write_text(json.dumps(cycle.to_dict(), indent=2, default=str))
    return out_dir


# ----- entry point used by CLI ---------------------------------------------


def orchestrate(
    ips_path: str | Path,
    limit: int | None = None,
    dry_run: bool = True,
    run_meta: bool = False,
) -> CycleResult:
    """Top-level entry: load IPS, build session, run cycle, save."""
    ips = load_ips(ips_path)
    errs = validate_ips(ips)
    if errs:
        raise ValueError(f"IPS validation failed:\n  - " + "\n  - ".join(errs))

    session = None
    if not dry_run:
        session = load_session_for_ips(ips, limit=limit)

    cycle = run_cycle(ips=ips, session=session, dry_run=dry_run)
    if run_meta:
        meta_result = run_meta_pass(ips=ips, session=session, dry_run=dry_run)
        cycle.agent_results["meta"] = meta_result

    save_cycle(cycle)
    return cycle
