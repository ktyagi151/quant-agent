"""Reviewer meta-agent: produce a senior-PM-style critique of prior research sessions.

This is a single long-context Anthropic API call, not an agent loop. The reviewer
reads every run's goal + final report + key thinking blocks, grades each run
across six capability dimensions, and synthesizes a meta-analysis.

Usage:
  from quant_agent.review import run_review, save_review
  result = run_review()          # auto-discovers runs under outputs/
  save_review(result)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

from .io_utils import outputs_dir


REVIEWER_SYSTEM = """You are a senior quant PM reviewing the work of a junior \
researcher who has run a series of experiments on a cross-sectional long/short \
pipeline. Your job is to produce a grounded, honest evaluation of their work \
across six capability dimensions, then synthesize what a future user of this \
agent should know.

# The dimensions
1. **Ideation** — Quality of ideas. Are features motivated by economic thinking \
and cited domain knowledge, or are they mechanical pandas tricks?
2. **Diagnosis** — When things fail, does the researcher correctly identify WHY? \
IC-quality, turnover-bound, regime-fit, redundancy, and tradability (smile-vs-slope) \
failures are all different; do they distinguish?
3. **Iteration** — Do successive steps build on prior findings, or does the \
researcher start fresh each turn?
4. **Self-correction** — Does the researcher catch their own errors in reasoning, \
update beliefs when evidence contradicts predictions, or revise mental models of \
the tools they're using?
5. **Strategic thinking** — Do they know when to pivot vs. tune? Recognize \
cost-bound situations? Identify root causes vs. treat symptoms?
6. **Communication** — Do final reports answer the actual question in a way a PM \
can act on? Are negative results delivered with the same rigor as positive ones? \
Are limitations acknowledged?

# Your deliverable (structure precisely)

## Per-run grades
For each run, a compact block:
  Run N — [one-phrase label]
  Goal (compressed to one sentence)
  Grades: Ideation X/5 | Diagnosis X/5 | Iteration X/5 | Self-correction X/5 | Strategic X/5 | Communication X/5
  Evidence: 1-2 sentences quoting or citing specific moments. Include both strengths AND weaknesses.

## Synthesis (the main payoff for the reviewing PM)
- **Consistent strengths**: dimensions the agent is reliably good at, with patterns.
- **Consistent weaknesses**: dimensions that need human review, with specific failure modes that recur.
- **What to trust the agent to do autonomously** (i.e., you'd ship the output without heavy review).
- **What to review / gate**: work categories where a human PM should read the full transcript before using results.
- **Recurring blind spots or tics**: idiosyncrasies of this agent's thinking (e.g., tendency to over-explain a single metric, tendency to cite X academic even when unrelated).
- **Single strongest example** of quant-flavored thinking across all runs — quote it.
- **Single weakest example** across all runs — quote it.

# Rules
- Be honest. 5/5 on everything is a fail of this evaluation; there is always something to critique.
- Cite specific quotes or behaviors as evidence — generic "good reasoning" is not acceptable.
- Keep the per-run grade blocks TIGHT. The synthesis is where the value is.
- You are not grading the pipeline or the outcome (did they beat the baseline?). You are grading the RESEARCH PROCESS.
- Negative results delivered well are valuable. An agent that fails thoughtfully is useful.
- If you identify a failure pattern, be specific about the triggering condition so a future user knows what to watch for.
"""


def _extract_run_summary(run_dir: Path, max_thinking_chars: int = 6000) -> dict:
    """Pull goal + final report + top thinking excerpts + tool sequence from a run dir."""
    goal_path = run_dir / "goal.txt"
    transcript_path = run_dir / "transcript.json"
    report_path = run_dir / "final_report.md"

    out: dict = {
        "run_id": run_dir.name,
        "goal": goal_path.read_text().strip() if goal_path.exists() else "(goal not recorded)",
        "final_report": report_path.read_text() if report_path.exists() else "(no final report saved)",
        "thinking_excerpt": "",
        "tool_sequence": [],
    }

    if transcript_path.exists():
        transcript = json.loads(transcript_path.read_text())
        thinking_blocks: list[str] = []
        for msg in transcript:
            for block in msg.get("content", []):
                btype = block.get("type")
                if btype == "thinking":
                    text = (block.get("text") or "").strip()
                    if text:
                        thinking_blocks.append(text)
                elif btype == "tool_use":
                    out["tool_sequence"].append(block.get("name", "?"))

        # Budget the thinking excerpt — keep the most recent ones up to the char cap
        # (the late-run thinking tends to capture the synthesis).
        combined = "\n\n---\n\n".join(thinking_blocks)
        if len(combined) > max_thinking_chars:
            combined = "[...earlier thinking truncated...]\n\n" + combined[-max_thinking_chars:]
        out["thinking_excerpt"] = combined

    return out


def build_review_prompt(run_summaries: list[dict]) -> str:
    """Render the per-run content into a single user-message prompt."""
    sections = []
    for i, s in enumerate(run_summaries, 1):
        sections.append(
            f"## Run {i} — directory `{s['run_id']}`\n\n"
            f"### Goal given to the agent\n{s['goal']}\n\n"
            f"### Tool call sequence ({len(s['tool_sequence'])} calls)\n"
            f"{', '.join(s['tool_sequence']) or '(none)'}\n\n"
            f"### Final report\n{s['final_report']}\n\n"
            f"### Selected thinking excerpts\n{s['thinking_excerpt'] or '(no thinking captured)'}"
        )
    return "Below are the runs, in chronological order.\n\n" + "\n\n---\n\n".join(sections) + (
        "\n\n---\n\nNow produce your per-run grades followed by the synthesis. "
        "Be honest and specific. Cite quotes. Identify patterns."
    )


def run_review(
    run_dirs: list[Path] | None = None,
    model: str = "claude-opus-4-7",
    max_tokens: int = 32000,
) -> dict:
    """Auto-discover research runs (or use provided list), call Claude, return result."""
    if run_dirs is None:
        out_root = outputs_dir()
        run_dirs = sorted(out_root.glob("research_*"))
    if not run_dirs:
        raise ValueError("no research_* directories found under outputs/")

    summaries = [_extract_run_summary(d) for d in run_dirs]
    user_prompt = build_review_prompt(summaries)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = Anthropic()

    # Opus 4.7: adaptive thinking + high effort. This is a deep-analysis task,
    # so let Claude think. Stream to respect timeout ceilings on long outputs.
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "high"},
        system=[{"type": "text", "text": REVIEWER_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for _ in stream.text_stream:
            pass
        final = stream.get_final_message()

    review_text = "\n".join(b.text for b in final.content if b.type == "text")
    thinking_text = "\n\n---\n\n".join(
        (b.thinking or "") for b in final.content if b.type == "thinking"
    )

    return {
        "review": review_text,
        "thinking": thinking_text,
        "run_dirs": [str(d) for d in run_dirs],
        "usage": {
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                final.usage, "cache_creation_input_tokens", 0
            ),
            "cache_read_input_tokens": getattr(
                final.usage, "cache_read_input_tokens", 0
            ),
        },
    }


def save_review(result: dict, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = outputs_dir() / f"review_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REVIEW.md").write_text(result["review"])
    if result.get("thinking"):
        (out_dir / "reviewer_thinking.md").write_text(result["thinking"])
    (out_dir / "meta.json").write_text(
        json.dumps(
            {"run_dirs": result["run_dirs"], "usage": result["usage"]},
            indent=2,
            default=str,
        )
    )
    return out_dir
