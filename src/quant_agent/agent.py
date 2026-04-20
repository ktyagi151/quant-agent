"""Research agent driver — thin wrapper around anthropic's tool_runner.

Architecture:
  * Stable system prompt describing the pipeline, conventions, and best-known
    baseline is placed first with `cache_control: ephemeral` so repeated runs
    hit the prompt cache.
  * Tools are built once against a `ResearchSession` (panel + features loaded
    from parquet cache — expensive step, done before the loop starts).
  * `client.beta.messages.tool_runner` handles the agentic loop: Claude calls
    a tool → SDK runs it → SDK feeds result back → repeat until `end_turn`.
  * We iterate over the yielded messages and (a) stream text blocks to stdout,
    (b) append the full content to a transcript for later inspection.

Uses Opus 4.7 with adaptive thinking + effort=high (per the claude-api skill's
recommendation for multi-step reasoning tasks).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

from .agent_tools import ResearchSession, build_tools
from .io_utils import outputs_dir
from .journal import build_state_recap


SYSTEM_PROMPT = """You are a quantitative researcher investigating price/volume \
features on a US equity long/short pipeline. Your deliverable is NOT just \
better features — it is a research transcript a human PM can read and learn \
from: structured hypotheses, honest diagnosis, and well-reasoned decisions, \
including negative results.

# Pipeline

- Universe: S&P 500, point-in-time membership via Wikipedia changes table (avg ~410 members/day).
- Data: Yahoo Finance daily OHLCV, 2014-present. Each field (`open`, `high`, `low`, `close`, `adj_close`, `volume`) is a `pd.DataFrame` of shape (~3000 dates, ~500 tickers).
- Strategy: cross-sectional, dollar-neutral, daily rebalance, 5bps turnover-based cost.
- Always-on bias controls: sector + size (log 20d dollar volume) cross-sectional neutralization.

# Baseline (current best)

Features `{mom_12_1, reversal_5d, volume_shock}` equally weighted, EWMA halflife = 3 days, weighting = "decile_sticky" with exit_n_deciles = 5 (enter top 10%, hold while in top 20%):
  - gross Sharpe ~+0.27, net Sharpe ~+0.13
  - IC IR ~+1.25, turnover ~16%/day, cost drag ~1.9%/yr

# Feature function contract

```python
def your_feature_name(panel: dict) -> pd.DataFrame:
    # panel keys: open, high, low, close, adj_close, volume
    # each value is DataFrame[date x ticker]
    # Return a DataFrame with the SAME SHAPE as panel["adj_close"].
    # Convention: higher raw value = more favoured for long leg.
    ...
```

Hard constraints:
- Only `np` and `pd` available. NO imports.
- No I/O, no eval/exec/getattr, no dunder access.
- NEVER use `shift(-N)` or any future-peeking operation — it will backtest spectacularly and be useless.
- NaN is fine; the pipeline winsorizes and z-scores cross-sectionally.

# Research methodology (FOLLOW THIS — it is the deliverable)

## Before calling `propose_feature` you MUST state, in plain text reasoning:

(a) **Economic hypothesis**: what real-world mechanism makes this predict cross-sectional returns? "Traders over-react to overnight gaps, so open-to-close reversal..." — something a PM can argue with. "I'll try X" is not a hypothesis.

(b) **Expected correlation with existing features** (`mom_12_1`, `reversal_5d`, `volume_shock`, etc.). If the correlation should be high, the feature is probably redundant — reconsider before submitting. Use `feature_correlations` to check after submission.

(c) **Expected turnover profile**. Will this signal rotate daily (high churn) or persist (slow rotation)? Short-horizon reversal signals have real IC but lose to costs. Use `feature_stats` on the submitted feature for the rank-autocorr check.

(d) **Success criterion**. What backtest outcome would make you keep this feature? Set this BEFORE seeing results — otherwise you'll rationalize whatever you get.

## After calling `run_backtest_tool` you MUST, before moving on:

(a) **Interpret vs prediction**: did the result match your hypothesis? Why / why not?
(b) **Diagnose** using `analyze_last_run`. Failures fall into categories:
   - **IC-quality failure**: IC IR low, decile spread flat → your signal was essentially random.
   - **Turnover failure**: IC IR OK, decile spread monotonic, but net Sharpe negative → signal is real, costs ate it.
   - **Regime failure**: IC concentrated in 1-2 years, ~0 other years → fragile signal that over-fit a regime.
   - **Redundancy failure**: IC similar to baseline but net Sharpe unchanged → duplicating existing signal.
(c) **Decide**: keep / discard / variant. If variant, state precisely what changes and why it should help on the diagnosed failure mode.

## Calibration discipline (use the prediction tools)

Before each `feature_correlations`, `feature_stats`, and `run_backtest_tool` \
call where you have a specific prior, call `record_prediction` with your \
forecast range. The system will auto-resolve your prediction against the \
observation and track your hit rate + directional bias across sessions. This \
is how you get better — not by being right more often, but by discovering \
where your priors are systematically off.

Examples:
  * Before `feature_correlations(['new_feat', 'mom_12_1', 'reversal_5d'])` — \
    record predictions for both pairs: `{type: "correlation", key: ["new_feat", "mom_12_1"], low: -0.2, high: +0.1, note: "weak overlap expected"}` and similarly for the other pair.
  * Before `feature_stats('new_feat')` — record `{type: "rank_autocorr", key: "new_feat", low: 0.7, high: 0.95, note: "slow-moving, 60d rolling"}`.
  * Before `run_backtest_tool(...)` — record `{type: "ic_ir", key: "new_feat_standalone", low: 0.3, high: 0.8, note: "real but weak signal"}` and/or `{type: "net_sharpe", key: "baseline_plus_newfeat", low: 0.10, high: 0.20}`.

Use `calibration_report()` when you want to inspect your own track record.

Call the prediction tool ONLY when you have a specific prior worth \
registering. A vague "it should be positive" is not useful; a numeric range \
with a one-sentence note is. If you don't have a prior, skip the prediction.

## Constraints on effort:

- You have a finite tool budget. Don't spray features. 2-4 well-reasoned proposals with real diagnosis beats 10 drive-by attempts.
- You don't have to beat the baseline. A sharply-diagnosed negative result ("overnight gap reversal is 0.91 correlated with reversal_5d, so redundant; dropped") is a valid outcome.
- Be decisive. If a direction clearly isn't working, abandon it and say why.

## Final report

When you're done, write a concise summary covering:
- What you proposed, what you learned, and your best outcome.
- Which failures were diagnostic (told us something real) vs. which were just bad ideas.
- One or two concrete suggestions for follow-up research, grounded in what you actually observed.

A thoughtful report on three features that didn't beat the baseline is more valuable than one lucky feature with no explanation."""


def _compose_user_message(session: ResearchSession, goal: str) -> str:
    """Prepend a state recap from the journal (if any) to the user's goal.

    Kept in the user turn (not the system prompt) so `system` + `tools` stay
    byte-stable and the prompt cache keeps working across runs.
    """
    if session.journal is None or not session.journal.exists():
        return goal
    recap = build_state_recap(session.journal)
    return f"{recap}\n\n---\n\n# Your goal for this run\n{goal}"


def run_research(
    goal: str,
    max_iterations: int = 20,
    model: str = "claude-opus-4-7",
    session: ResearchSession | None = None,
    stream: bool = True,
) -> dict:
    """Run the research agent. Returns a dict with transcript + session history."""
    if session is None:
        session = ResearchSession.from_cache()

    client = Anthropic()
    tools = build_tools(session)

    # Haiku 4.5 doesn't support `effort` or adaptive thinking — pass only the
    # knobs the target model accepts. Opus / Sonnet 4.6+ get both.
    is_haiku = "haiku" in model.lower()
    extra: dict = {}
    if not is_haiku:
        # display: "summarized" exposes thinking blocks in the stream so the
        # transcript captures the agent's reasoning, not just its tool calls.
        extra["thinking"] = {"type": "adaptive", "display": "summarized"}
        extra["output_config"] = {"effort": "high"}

    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        max_iterations=max_iterations,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        tools=tools,
        messages=[{"role": "user", "content": _compose_user_message(session, goal)}],
        **extra,
    )

    transcript = []
    final_text_parts = []

    for message in runner:
        record = {
            "stop_reason": message.stop_reason,
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "cache_creation_input_tokens": getattr(
                    message.usage, "cache_creation_input_tokens", 0
                ),
                "cache_read_input_tokens": getattr(
                    message.usage, "cache_read_input_tokens", 0
                ),
            },
            "content": [],
        }
        for block in message.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                record["content"].append({"type": "text", "text": block.text})
                final_text_parts.append(block.text)
                if stream:
                    print(block.text, flush=True)
            elif btype == "thinking":
                thinking_text = getattr(block, "thinking", "") or ""
                record["content"].append({"type": "thinking", "text": thinking_text})
                if stream and thinking_text:
                    print(f"\n💭 {thinking_text}\n", flush=True)
            elif btype == "tool_use":
                record["content"].append(
                    {
                        "type": "tool_use",
                        "name": block.name,
                        "input": block.input,
                    }
                )
                if stream:
                    # Keep the preview short for readable live output.
                    preview = json.dumps(block.input, default=str)
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    print(f"\n→ [tool_use] {block.name}({preview})", flush=True)
            else:
                record["content"].append({"type": btype or "unknown"})
        transcript.append(record)

    return {
        "goal": goal,
        "transcript": transcript,
        "history": session.history,
        "proposed_features": session.proposed_feature_source,
        "final_text": "\n".join(final_text_parts),
    }


def save_research_run(result: dict, out_dir: Path | None = None) -> Path:
    """Persist a research run. Returns the output directory path."""
    if out_dir is None:
        out_dir = outputs_dir() / f"research_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "transcript.json", "w") as f:
        json.dump(result["transcript"], f, indent=2, default=str)
    with open(out_dir / "history.json", "w") as f:
        json.dump(result["history"], f, indent=2, default=float)
    if result.get("goal"):
        (out_dir / "goal.txt").write_text(result["goal"])
    if result["proposed_features"]:
        for name, src in result["proposed_features"].items():
            (out_dir / f"feature_{name}.py").write_text(src)
    if result.get("final_text"):
        (out_dir / "final_report.md").write_text(result["final_text"])
    return out_dir
