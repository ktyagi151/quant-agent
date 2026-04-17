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


SYSTEM_PROMPT = """You are a quantitative researcher working on a US equity \
long/short pipeline. Your job: propose new price/volume features and test \
whether they improve net Sharpe against the current baseline.

# Pipeline overview
- Universe: S&P 500, point-in-time membership reconstructed from Wikipedia.
- Data: Yahoo Finance daily OHLCV, 2014-present, cached locally as parquet.
- Panel shape: each field (`open`, `high`, `low`, `close`, `adj_close`, `volume`) is a
  pandas DataFrame with dates (rows) x tickers (columns). ~3000 dates, ~500 tickers.
- Strategy: cross-sectional, dollar-neutral, decile long/short with 5bps cost, daily rebalance.
- Always-on bias controls: sector + size (log 20d dollar volume) neutralization.

# Baseline (current best)
Stack: features {mom_12_1, reversal_5d, volume_shock} equally weighted, EWMA \
halflife = 3 days, weighting = "decile_sticky" with exit_n_deciles = 5 \
(enter top 10%, hold while in top 20%). This hits:
  - gross Sharpe ~+0.27
  - net Sharpe ~+0.13
  - IC IR ~+1.25
  - turnover ~16%/day
  - cost drag ~1.9%/yr

# Feature function contract
```python
def your_feature_name(panel: dict) -> pd.DataFrame:
    # panel has keys: open, high, low, close, adj_close, volume
    # each is DataFrame[date x ticker]
    # Return DataFrame with the SAME SHAPE AS panel["adj_close"].
    # Convention: higher raw value = more favoured for long leg.
    ...
```

Constraints on code:
- Only `np` and `pd` are available. NO imports.
- No I/O, no eval/exec, no getattr. Pure pandas/numpy math.
- NaN is fine — the combine step winsorizes and z-scores per row.

# What makes a good feature
- Economic intuition. Price reversal, momentum (at various horizons), volume \
patterns, volatility signals, overnight-vs-intraday splits, high-low range \
relative to close, volume-weighted returns, etc.
- Not redundant with existing features. If you propose yet another momentum \
variant it will correlate with `mom_12_1`. The `neutralize` step handles size \
and sector automatically, so don't duplicate those.
- Reasonable turnover. Very short-horizon signals (like 1-day reversal) have \
IC but pay enormous cost. Think about signal persistence.
- Handle edge cases: zero volume, rolling windows, look-ahead (never use \
`shift(-N)` or future data).

# Workflow
1. Call `list_features` to see what's registered.
2. Propose a feature: explain the intuition, write the function, submit via \
   `propose_feature`. If it's rejected, read the error and fix.
3. Backtest it: call `run_backtest_tool` with your feature added to the baseline \
   weights. Tune weight (start small, e.g. 0.5 vs 1.0 for baseline features).
4. Look at the result. Did net Sharpe improve? What about IC IR (rank quality) \
   vs turnover (whether costs eat the alpha)? Iterate if promising, move on if not.
5. After testing 2-4 features, write a concise final report: which features \
   helped, which didn't, and why. Include before/after net Sharpe.

Be decisive — don't re-test obvious variants. Be honest about what didn't work; \
negative results matter for the user."""


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

    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        max_iterations=max_iterations,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        tools=tools,
        messages=[{"role": "user", "content": _compose_user_message(session, goal)}],
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
                record["content"].append(
                    {"type": "thinking", "summary": getattr(block, "thinking", "")[:200]}
                )
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
    if result["proposed_features"]:
        for name, src in result["proposed_features"].items():
            (out_dir / f"feature_{name}.py").write_text(src)
    if result.get("final_text"):
        (out_dir / "final_report.md").write_text(result["final_text"])
    return out_dir
