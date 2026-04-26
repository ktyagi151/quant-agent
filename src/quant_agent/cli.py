"""Command-line entry points."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
import yaml

from . import data as data_mod
from . import features as feat_mod
from . import metrics as met
from . import neutralize as neut_mod
from . import signals as sig_mod
from . import universe as uni_mod
from .backtest import run_backtest
from .io_utils import outputs_dir


@click.group()
def cli():
    """Quant Agent CLI."""


@cli.command("fetch-universe")
@click.option("--refresh/--no-refresh", default=True)
def fetch_universe(refresh: bool) -> None:
    tickers = uni_mod.get_sp500_tickers(refresh=refresh)
    click.echo(f"{len(tickers)} tickers (first 10): {tickers[:10]}")


@cli.command("fetch-data")
@click.option("--start", default="2014-01-01")
@click.option("--end", default="today")
@click.option("--limit", type=int, default=None, help="Only fetch first N tickers (debug).")
def fetch_data(start: str, end: str, limit: int | None) -> None:
    tickers = uni_mod.get_sp500_tickers(refresh=False)
    if limit:
        tickers = tickers[:limit]
    raw = data_mod.fetch_ohlcv(tickers, start, end)
    n_ok = sum(1 for df in raw.values() if not df.empty)
    click.echo(f"fetched OHLCV for {n_ok}/{len(tickers)} tickers, window [{start}, {end}]")


def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@cli.command("run-sample")
@click.option("--config", default="configs/default.yaml")
@click.option("--limit", type=int, default=None)
@click.option("--refresh/--no-refresh", default=False, help="Refresh cache before backtest.")
def run_sample(config: str, limit: int | None, refresh: bool) -> None:
    cfg = _load_config(config)
    start, end = cfg["start"], cfg["end"]

    tickers = uni_mod.get_sp500_tickers(refresh=False)
    if limit:
        tickers = tickers[:limit]

    click.echo(f"universe: {len(tickers)} tickers")
    raw = (
        data_mod.fetch_ohlcv(tickers, start, end)
        if refresh
        else data_mod.load_panel(tickers, start, end)
    )
    panel = data_mod.to_wide_panel(raw)

    feat_names = list(cfg["features"].keys())
    # Always also compute dollar_vol_20d for liquidity filter.
    all_feats = list(dict.fromkeys(feat_names + ["dollar_vol_20d"]))
    feats = feat_mod.compute_features(panel, all_feats)

    signal = sig_mod.combine(feats, cfg["features"])
    if "liquidity" in cfg and "dollar_vol_20d" in feats:
        signal = sig_mod.apply_liquidity_filter(
            signal, feats["dollar_vol_20d"], cfg["liquidity"]["min_dollar_vol_usd"]
        )

    # Point-in-time membership mask (reduces survivorship bias).
    mem_cfg = cfg.get("membership", {})
    if mem_cfg.get("point_in_time", False):
        mem = uni_mod.build_membership_matrix(start, end)
        # Align mask to signal grid; tickers absent from mem → treat as not-member (NaN out).
        mask = mem.reindex(index=signal.index, columns=signal.columns).fillna(False)
        signal = signal.where(mask)
        n_members_avg = float(mask.sum(axis=1).mean())
        click.echo(f"membership mask applied — avg {n_members_avg:.1f} tickers/day")

    # Optional EWMA smoothing (reduces turnover by damping signal churn).
    smooth_cfg = cfg.get("smoothing", {})
    hl = smooth_cfg.get("halflife_days", 0)
    if hl and hl > 0:
        signal = sig_mod.smooth_ewma(signal, halflife_days=hl)
        click.echo(f"smoothed: EWMA halflife={hl}d")

    # Sector + size neutralization.
    neut_cfg = cfg.get("neutralize", {})
    sectors = None
    size_df = None
    if neut_cfg.get("sector", False):
        snap = uni_mod.load_latest_snapshot()
        if snap is not None and "sector" in snap.columns:
            sectors = snap.set_index("ticker")["sector"]
    if neut_cfg.get("size", False) and "dollar_vol_20d" in feats:
        size_df = feats["dollar_vol_20d"]
    if sectors is not None or size_df is not None:
        signal = neut_mod.neutralize(signal, sectors=sectors, size=size_df)
        tags = []
        if sectors is not None:
            tags.append("sector")
        if size_df is not None:
            tags.append("size")
        click.echo(f"neutralized: {', '.join(tags)}")

    prices = panel[cfg["backtest"].get("return_field", "adj_close")]
    bt_cfg = cfg["backtest"]
    res = run_backtest(
        signal,
        prices,
        n_deciles=bt_cfg["n_deciles"],
        cost_bps=bt_cfg["cost_bps"],
        weighting=bt_cfg.get("weighting", "decile"),
        exit_n_deciles=bt_cfg.get("exit_n_deciles"),
    )
    click.echo(f"weighting: {bt_cfg.get('weighting', 'decile')}")

    ic = met.information_coefficient(signal, prices.pct_change().shift(-1))
    summary = met.summary(res.returns, turnover=res.turnover)
    summary.update(met.ic_summary(ic))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = outputs_dir() / f"sample_run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    res.returns.to_frame("net").assign(gross=res.gross_returns, turnover=res.turnover).to_parquet(
        run_dir / "returns.parquet"
    )
    res.weights.to_parquet(run_dir / "weights.parquet")
    res.per_decile_returns.to_parquet(run_dir / "per_decile_returns.parquet")
    ic.to_frame().to_parquet(run_dir / "ic.parquet")
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # Simple equity curve PNG.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eq = (1 + res.returns.fillna(0)).cumprod()
        fig, ax = plt.subplots(figsize=(10, 4))
        eq.plot(ax=ax, title=f"Net equity (cost_bps={res.cost_bps})")
        ax.set_ylabel("equity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "equity.png", dpi=120)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        click.echo(f"plot skipped: {e}")

    click.echo(json.dumps(summary, indent=2, default=float))
    click.echo(f"outputs: {run_dir}")


def _run_from_cfg(cfg: dict, limit: int | None = None) -> dict:
    """Run a single backtest from a config dict, return summary + key diagnostics."""
    start, end = cfg["start"], cfg["end"]
    tickers = uni_mod.get_sp500_tickers(refresh=False)
    if limit:
        tickers = tickers[:limit]

    raw = data_mod.load_panel(tickers, start, end)
    panel = data_mod.to_wide_panel(raw)

    feat_names = list(cfg["features"].keys())
    all_feats = list(dict.fromkeys(feat_names + ["dollar_vol_20d"]))
    feats = feat_mod.compute_features(panel, all_feats)

    signal = sig_mod.combine(feats, cfg["features"])
    if "liquidity" in cfg and "dollar_vol_20d" in feats:
        signal = sig_mod.apply_liquidity_filter(
            signal, feats["dollar_vol_20d"], cfg["liquidity"]["min_dollar_vol_usd"]
        )

    if cfg.get("membership", {}).get("point_in_time", False):
        mem = uni_mod.build_membership_matrix(start, end)
        mask = mem.reindex(index=signal.index, columns=signal.columns).fillna(False)
        signal = signal.where(mask)

    hl = cfg.get("smoothing", {}).get("halflife_days", 0)
    if hl and hl > 0:
        signal = sig_mod.smooth_ewma(signal, halflife_days=hl)

    neut_cfg = cfg.get("neutralize", {})
    sectors = None
    size_df = None
    if neut_cfg.get("sector", False):
        snap = uni_mod.load_latest_snapshot()
        if snap is not None and "sector" in snap.columns:
            sectors = snap.set_index("ticker")["sector"]
    if neut_cfg.get("size", False) and "dollar_vol_20d" in feats:
        size_df = feats["dollar_vol_20d"]
    if sectors is not None or size_df is not None:
        signal = neut_mod.neutralize(signal, sectors=sectors, size=size_df)

    prices = panel[cfg["backtest"].get("return_field", "adj_close")]
    bt_cfg = cfg["backtest"]
    res = run_backtest(
        signal,
        prices,
        n_deciles=bt_cfg["n_deciles"],
        cost_bps=bt_cfg["cost_bps"],
        weighting=bt_cfg.get("weighting", "decile"),
        exit_n_deciles=bt_cfg.get("exit_n_deciles"),
    )
    ic = met.information_coefficient(signal, prices.pct_change().shift(-1))
    summary = met.summary(res.returns, turnover=res.turnover)
    summary.update(met.ic_summary(ic))
    # Add gross metrics too.
    gross_summary = met.summary(res.gross_returns)
    summary["gross_sharpe"] = gross_summary["sharpe"]
    summary["gross_ann_return"] = gross_summary["ann_return"]
    summary["cost_drag_ann"] = gross_summary["ann_return"] - summary["ann_return"]
    return summary


def _run_one(cfg_path: Path, limit: int | None = None) -> dict:
    return _run_from_cfg(_load_config(cfg_path), limit=limit)


@cli.command("compare")
@click.option(
    "--configs-dir",
    default="configs/variants",
    help="Directory of YAML variant configs to compare.",
)
@click.option("--limit", type=int, default=None)
def compare(configs_dir: str, limit: int | None) -> None:
    """Run every variant config in `configs-dir` and print a comparison table."""
    paths = sorted(Path(configs_dir).glob("*.yaml"))
    if not paths:
        raise click.ClickException(f"no configs found in {configs_dir}")

    rows = []
    for p in paths:
        click.echo(f"\n=== running {p.name} ===")
        s = _run_one(p, limit=limit)
        s["variant"] = p.stem
        rows.append(s)

    df = pd.DataFrame(rows).set_index("variant")
    cols = [
        "gross_ann_return",
        "gross_sharpe",
        "cost_drag_ann",
        "ann_return",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "ic_mean",
        "ic_ir",
    ]
    cols = [c for c in cols if c in df.columns]
    disp = df[cols].copy()

    # Format.
    pct_cols = {"gross_ann_return", "ann_return", "cost_drag_ann", "max_drawdown", "avg_turnover"}
    for c in disp.columns:
        if c in pct_cols:
            disp[c] = disp[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
        else:
            disp[c] = disp[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—")

    click.echo("\n=== comparison ===")
    click.echo(disp.to_string())

    # Persist.
    out_dir = outputs_dir() / f"compare_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "comparison.parquet")
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(df.reset_index().to_dict(orient="records"), f, indent=2, default=float)
    click.echo(f"\nsaved: {out_dir}")


@cli.command("orchestrate")
@click.option("--ips", "ips_path", default="configs/ips/default.yaml", help="IPS YAML file.")
@click.option("--limit", type=int, default=None, help="Limit universe size (live runs only).")
@click.option("--dry-run/--live", default=True,
              help="Dry-run renders prompts without API calls (default). --live invokes Anthropic.")
@click.option("--meta/--no-meta", default=False, help="Also run the meta-agent pass.")
@click.option(
    "--cheap",
    is_flag=True,
    default=False,
    help="Force all agents to claude-haiku-4-5 with tight max_iter (for budget-bound smoke tests).",
)
@click.option(
    "--max-iter",
    type=int,
    default=None,
    help="Override max_iterations on every agent (e.g. 3 for a tight smoke test).",
)
def orchestrate(ips_path: str, limit: int | None, dry_run: bool, meta: bool,
                cheap: bool, max_iter: int | None) -> None:
    """Run one investment cycle through the multi-agent pipeline.

    Default is --dry-run: every agent's system prompt is rendered against
    the IPS, but no Anthropic call is made. Useful for inspecting wiring
    and prompt content without spending budget. Use --live to actually
    invoke the agents.

    --cheap: forces every agent to claude-haiku-4-5 (~5× cheaper than Opus)
    and tight max_tokens. Useful for under-$0.20 smoke tests against a real
    API key.
    """
    from .orchestrator import orchestrate as _orch
    from . import agents as ag

    # Apply runtime overrides to specs.
    if cheap or max_iter is not None:
        for spec in [ag.alpha_agent_spec, ag.portfolio_agent_spec,
                     ag.cost_risk_agent_spec, ag.critic_agent_spec, ag.meta_agent_spec]:
            if cheap:
                spec.model = "claude-haiku-4-5"
                spec.use_thinking = False    # Haiku doesn't support adaptive thinking
                spec.max_tokens = min(spec.max_tokens, 4000)
            if max_iter is not None:
                spec.max_iterations = max_iter

    cycle = _orch(ips_path=ips_path, limit=limit, dry_run=dry_run, run_meta=meta)
    click.echo(f"\ncycle: {cycle.cycle_id}")
    click.echo(f"verdict: {cycle.final_verdict}")
    if cycle.notes:
        click.echo(f"notes: {cycle.notes}")

    total_cost = 0.0
    for name, r in cycle.agent_results.items():
        if dry_run:
            preview = r.outputs.get("system_prompt_preview", "")
            chars = r.outputs.get("system_prompt_chars", 0)
            click.echo(f"\n  [{name}] system prompt: {chars} chars")
            if preview:
                click.echo(f"    preview: {preview[:120]}...")
        else:
            u = r.usage
            in_t = u.get("input_tokens", 0)
            out_t = u.get("output_tokens", 0)
            cache_r = u.get("cache_read_input_tokens", 0)
            # Cost depends on model — check by spec lookup.
            spec = next((s for s in [ag.alpha_agent_spec, ag.portfolio_agent_spec,
                                     ag.cost_risk_agent_spec, ag.critic_agent_spec, ag.meta_agent_spec]
                         if s.name == name), None)
            if spec and "haiku" in spec.model.lower():
                cost = in_t/1e6*1.0 + out_t/1e6*5.0 + cache_r/1e6*0.1
            else:
                cost = in_t/1e6*5.0 + out_t/1e6*25.0 + cache_r/1e6*0.5
            total_cost += cost
            click.echo(f"\n  [{name}] success={r.success} input={in_t} output={out_t} cache_read={cache_r} cost=${cost:.4f}")
            if r.error:
                click.echo(f"    error: {r.error}")

    if not dry_run:
        click.echo(f"\nTotal estimated cost: ${total_cost:.4f}")


@cli.command("review")
@click.option(
    "--runs",
    default=None,
    help="Comma-separated run directories (paths). If omitted, auto-discover all research_* under outputs/.",
)
@click.option("--model", default="claude-opus-4-7", help="Reviewer model.")
def review(runs: str | None, model: str) -> None:
    """Produce a senior-PM-style critique of prior research sessions.

    A single long-context Anthropic call. The reviewer reads each run's goal,
    final report, and thinking excerpts; grades across six capability axes;
    and synthesizes what a future user should trust vs double-check.
    """
    import os
    from pathlib import Path

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise click.ClickException("ANTHROPIC_API_KEY not set in environment.")

    from .review import run_review, save_review

    run_dirs = None
    if runs:
        run_dirs = [Path(p.strip()) for p in runs.split(",") if p.strip()]
        for p in run_dirs:
            if not p.exists():
                raise click.ClickException(f"run directory not found: {p}")

    click.echo(f"running review (model={model})...")
    result = run_review(run_dirs=run_dirs, model=model)
    out = save_review(result)
    click.echo(f"saved: {out}")
    u = result["usage"]
    click.echo(
        f"usage: input={u['input_tokens']}, output={u['output_tokens']}, "
        f"cache_read={u['cache_read_input_tokens']}"
    )


@cli.command("research")
@click.option("--goal", required=True, help="What you want the agent to investigate.")
@click.option("--max-iter", default=20, type=int, help="Max tool-use iterations.")
@click.option("--model", default="claude-opus-4-7", help="Anthropic model ID.")
@click.option(
    "--cheap",
    is_flag=True,
    default=False,
    help="Override --model with claude-haiku-4-5 for ~5x cheaper iteration (lower intelligence ceiling).",
)
@click.option("--limit", type=int, default=None, help="Limit universe size (for fast iteration).")
@click.option(
    "--fresh",
    is_flag=True,
    default=False,
    help="Ignore the research journal — start from baseline only (for ablations).",
)
def research(goal: str, max_iter: int, model: str, cheap: bool, limit: int | None, fresh: bool) -> None:
    """Run the LLM research agent against the cached pipeline.

    Requires ANTHROPIC_API_KEY in the environment. The agent can propose new
    feature functions (AST-sandboxed exec) and backtest them against the
    current best baseline. Accepted features and run history persist to
    `data/research/` by default; use --fresh to ignore the journal for a run.
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise click.ClickException("ANTHROPIC_API_KEY not set in environment.")

    if cheap:
        model = "claude-haiku-4-5"

    from .agent import run_research, save_research_run
    from .agent_tools import ResearchSession

    click.echo(f"loading research session (limit={limit}, fresh={fresh}, model={model})...")
    session = ResearchSession.from_cache(limit=limit, journal=False if fresh else True)
    click.echo(
        f"session ready: {len(session.panel['adj_close'].columns)} tickers, "
        f"{len(session.panel['adj_close'])} dates, "
        f"{len(session.feature_fns)} features registered."
    )
    if session.journal_load_warnings:
        click.echo("journal reload warnings:")
        for w in session.journal_load_warnings:
            click.echo(f"  - {w}")
    if session.journal is not None and session.journal.exists():
        best = session.journal.best()
        if best is not None:
            click.echo(
                f"journal: {session.journal.total_runs()} prior runs, "
                f"current best net sharpe "
                f"{best.get('summary', {}).get('sharpe', float('nan')):+.3f}"
            )

    result = run_research(goal, max_iterations=max_iter, model=model, session=session)
    out = save_research_run(result)
    click.echo(f"\n\nrun saved: {out}")
    if result["history"]:
        click.echo(f"backtests run this session: {len(result['history'])}")
        best = max(result["history"], key=lambda r: r.get("sharpe", float("-inf")))
        click.echo(
            f"best net sharpe this session: {best.get('sharpe'):+.3f} "
            f"with {best.get('config')}"
        )


@cli.group("journal")
def journal_cmd() -> None:
    """Inspect or clear the research journal."""


@journal_cmd.command("list")
@click.option("--top", default=5, type=int, help="Show top-N runs by net Sharpe.")
def journal_list(top: int) -> None:
    """Show persisted features, best run, and top-N runs."""
    from .journal import Journal

    j = Journal.default()
    if not j.exists():
        click.echo("No journal at data/research/ yet.")
        return

    feats = j.all_feature_metadata()
    click.echo(f"Persisted features: {len(feats)}")
    for m in feats:
        bs = m.get("best_sharpe")
        bs_str = f"{bs:+.3f}" if bs is not None else "—"
        click.echo(
            f"  - {m['name']:<30} best_sharpe={bs_str}  added={m.get('added_at', '—')}"
        )

    best = j.best()
    if best is not None:
        s = best.get("summary", {})
        click.echo(
            f"\nBest run: {best['run_id']} ({best['timestamp']})"
            f" net_sharpe={s.get('sharpe', 0):+.3f}"
            f" gross={s.get('gross_sharpe', 0):+.3f}"
            f" ic_ir={s.get('ic_ir', 0):+.3f}"
            f" turnover={s.get('avg_turnover', 0):.2%}"
        )

    top_runs = j.top_runs(top)
    if top_runs:
        click.echo(f"\nTop {len(top_runs)} runs by net Sharpe:")
        for r in top_runs:
            s = r["summary"]
            click.echo(
                f"  {r['run_id']}  sharpe={s.get('sharpe', 0):+.3f}"
                f"  ic_ir={s.get('ic_ir', 0):+.3f}"
                f"  cfg={r['config']}"
            )

    click.echo(f"\nTotal runs: {j.total_runs()}")


@journal_cmd.command("clear")
@click.confirmation_option(
    prompt="This will delete all persisted features and run history. Continue?"
)
def journal_clear() -> None:
    """Wipe the journal. Irreversible."""
    from .journal import Journal

    j = Journal.default()
    j.clear()
    click.echo(f"cleared: {j.root}")


@cli.command("grid")
@click.option("--base", default="configs/default.yaml", help="Base config (all settings except the grid axes).")
@click.option("--halflives", default="0,2,3,5,10", help="Comma-separated EWMA halflife days. 0 = off.")
@click.option("--exits", default="off,7,5,3", help="Comma-separated exit_n_deciles. 'off' = hard decile (no stickiness).")
@click.option("--limit", type=int, default=None)
def grid(base: str, halflives: str, exits: str, limit: int | None) -> None:
    """Run a (halflife × sticky exit band) grid and print pivot tables.

    Combines v3-style smoothing and v4-style hysteresis. Each cell is a full
    backtest on the current cache; runtime ~20-30s per cell.
    """
    base_cfg = _load_config(base)
    hl_values = [int(x.strip()) for x in halflives.split(",") if x.strip()]
    exit_values: list[int | None] = []
    for tok in exits.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        exit_values.append(None if tok in {"off", "none", "hard"} else int(tok))

    rows = []
    total = len(hl_values) * len(exit_values)
    for i, hl in enumerate(hl_values):
        for j, ex in enumerate(exit_values):
            cfg = _deepcopy_cfg(base_cfg)
            cfg.setdefault("smoothing", {})["halflife_days"] = hl
            bt = cfg.setdefault("backtest", {})
            if ex is None:
                bt["weighting"] = "decile"
                bt.pop("exit_n_deciles", None)
            else:
                bt["weighting"] = "decile_sticky"
                bt["exit_n_deciles"] = ex

            tag = f"hl={hl}d,exit={'off' if ex is None else ex}"
            click.echo(f"[{i * len(exit_values) + j + 1}/{total}] {tag}")
            s = _run_from_cfg(cfg, limit=limit)
            s["halflife"] = hl
            s["exit"] = "off" if ex is None else ex
            rows.append(s)

    df = pd.DataFrame(rows)

    # Pivots.
    ex_order = ["off" if e is None else e for e in exit_values]

    def _pivot(metric: str, fmt) -> pd.DataFrame:
        p = df.pivot(index="halflife", columns="exit", values=metric)
        p = p.reindex(index=hl_values, columns=ex_order)
        return p.map(lambda v: fmt(v) if pd.notna(v) else "—")

    pct = lambda v: f"{v:+.2%}"
    flt = lambda v: f"{v:+.3f}"

    click.echo("\n=== net Sharpe (primary) ===")
    click.echo(_pivot("sharpe", flt).to_string())
    click.echo("\n=== gross Sharpe ===")
    click.echo(_pivot("gross_sharpe", flt).to_string())
    click.echo("\n=== avg turnover ===")
    click.echo(_pivot("avg_turnover", pct).to_string())
    click.echo("\n=== max drawdown ===")
    click.echo(_pivot("max_drawdown", pct).to_string())
    click.echo("\n=== IC IR ===")
    click.echo(_pivot("ic_ir", flt).to_string())

    out_dir = outputs_dir() / f"grid_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Cast `exit` to string ("off" mixed with ints → arrow rejects).
    df_save = df.copy()
    df_save["exit"] = df_save["exit"].astype(str)
    df_save.to_parquet(out_dir / "grid.parquet")
    with open(out_dir / "grid.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, default=float)
    click.echo(f"\nsaved: {out_dir}")


def _deepcopy_cfg(cfg: dict) -> dict:
    import copy

    return copy.deepcopy(cfg)


if __name__ == "__main__":
    cli()
