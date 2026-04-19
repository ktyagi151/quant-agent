"""Tools exposed to the LLM research agent.

Design:
  * `ResearchSession` holds the (expensive-to-build) state once: wide panel,
    feature cache, membership mask, sectors, baseline backtest config.
  * `build_tools(session)` returns a list of `@beta_tool` callables that close
    over the session. These are what we pass to `client.beta.messages.tool_runner`.

The agent can:
  * `list_features` — see what's already in the registry.
  * `propose_feature` — submit a Python function (AST-scanned, sandboxed).
  * `run_backtest` — specify feature weights + knobs; get summary metrics back.

Keeping the tool surface tiny matters: every tool schema sits in the cached
prefix, and a bloated surface dilutes the agent's attention.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from anthropic import beta_tool

from . import data as data_mod
from . import features as feat_mod
from . import metrics as met
from . import neutralize as neut_mod
from . import signals as sig_mod
from . import universe as uni_mod
from .backtest import run_backtest
from .journal import Journal
from .sandbox import UnsafeCodeError, exec_feature


@dataclass
class RunArtifacts:
    """Per-run DataFrames kept in memory so the agent can inspect its own runs."""

    run_id: str | None
    signal: pd.DataFrame
    fwd_ret: pd.DataFrame
    weights: pd.DataFrame
    net_returns: pd.Series
    gross_returns: pd.Series
    per_decile: pd.DataFrame
    summary: dict
    config: dict


@dataclass
class ResearchSession:
    """Holds pipeline state the agent repeatedly hits."""

    panel: dict[str, pd.DataFrame]
    feature_fns: dict[str, Callable]
    feature_cache: dict[str, pd.DataFrame]
    sectors: pd.Series | None
    membership_mask: pd.DataFrame | None
    liquidity_threshold: float
    cost_bps: float
    n_deciles: int
    history: list[dict] = field(default_factory=list)
    proposed_feature_source: dict[str, str] = field(default_factory=dict)
    journal: Journal | None = None
    journal_load_warnings: list[str] = field(default_factory=list)
    # Ring buffer of recent run artifacts (signal + returns etc.) so
    # `analyze_last_run` and friends don't need to recompute.
    recent_runs: list["RunArtifacts"] = field(default_factory=list)
    recent_runs_max: int = 5

    @classmethod
    def from_cache(
        cls,
        start: str = "2014-01-01",
        end: str = "today",
        limit: int | None = None,
        liquidity_threshold: float = 5_000_000,
        cost_bps: float = 5.0,
        n_deciles: int = 10,
        journal: Journal | None | bool = True,
    ) -> "ResearchSession":
        """Build a session from the existing parquet cache and universe snapshot.

        `journal`:
          * True (default) → load the default journal at `data/research/`
          * False → do not use a journal (fresh session, nothing persisted)
          * Journal instance → use the provided journal
        """
        tickers = uni_mod.get_sp500_tickers(refresh=False)
        if limit:
            tickers = tickers[:limit]
        raw = data_mod.load_panel(tickers, start, end)
        panel = data_mod.to_wide_panel(raw)

        mem = uni_mod.build_membership_matrix(start, end)

        sectors = None
        snap = uni_mod.load_latest_snapshot()
        if snap is not None and "sector" in snap.columns:
            sectors = snap.set_index("ticker")["sector"]

        # Precompute baseline features (they're cheap and always needed).
        baseline_feats = feat_mod.compute_features(
            panel,
            ["mom_12_1", "reversal_5d", "volume_shock", "dollar_vol_20d", "vol_21d", "amihud_20d"],
        )

        # Resolve journal parameter.
        if journal is True:
            j: Journal | None = Journal.default()
        elif journal is False or journal is None:
            j = None
        else:
            j = journal

        feature_fns = dict(feat_mod.FEATURES)
        feature_cache = dict(baseline_feats)
        proposed: dict[str, str] = {}
        warnings: list[str] = []

        if j is not None and j.exists():
            loaded = j.load_features()
            for lf in loaded:
                feature_fns[lf.name] = lf.fn
                proposed[lf.name] = lf.source
                # Precompute the loaded feature once so the first backtest doesn't pay the cost.
                try:
                    feature_cache[lf.name] = lf.fn(panel)
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"{lf.name}: precompute raised {type(e).__name__}: {e}")
                    # Drop this feature from the registry since it's broken in this panel.
                    feature_fns.pop(lf.name, None)
                    proposed.pop(lf.name, None)
            warnings.extend(j.load_warnings)

        return cls(
            panel=panel,
            feature_fns=feature_fns,
            feature_cache=feature_cache,
            sectors=sectors,
            membership_mask=mem,
            liquidity_threshold=liquidity_threshold,
            cost_bps=cost_bps,
            n_deciles=n_deciles,
            proposed_feature_source=proposed,
            journal=j,
            journal_load_warnings=warnings,
        )

    # ------------------------------------------------------------------

    def _feature(self, name: str) -> pd.DataFrame:
        if name in self.feature_cache:
            return self.feature_cache[name]
        if name not in self.feature_fns:
            raise KeyError(f"unknown feature: {name}")
        df = self.feature_fns[name](self.panel)
        self.feature_cache[name] = df
        return df

    def register_feature(self, name: str, source: str, description: str = "") -> None:
        fn = exec_feature(source, expected_name=name)
        # Smoke-test that it executes and returns a DataFrame with the right shape.
        try:
            df = fn(self.panel)
        except Exception as e:  # noqa: BLE001
            raise UnsafeCodeError(f"feature raised at runtime: {e}") from e
        if not isinstance(df, pd.DataFrame):
            raise UnsafeCodeError(
                f"feature must return pd.DataFrame, got {type(df).__name__}"
            )
        ref_shape = self.panel["adj_close"].shape
        if df.shape != ref_shape:
            raise UnsafeCodeError(
                f"feature returned shape {df.shape}, expected {ref_shape} "
                f"(dates x tickers matching adj_close)"
            )
        self.feature_fns[name] = fn
        self.feature_cache[name] = df
        self.proposed_feature_source[name] = source
        if self.journal is not None:
            self.journal.save_feature(name, source, description)

    def run(
        self,
        feature_weights: dict[str, float],
        halflife_days: float,
        weighting: str,
        exit_n_deciles: int | None,
    ) -> dict:
        if not feature_weights:
            raise ValueError("feature_weights must not be empty")
        for name in feature_weights:
            if name not in self.feature_fns:
                raise KeyError(
                    f"unknown feature: {name}. Use list_features or propose_feature first."
                )

        feats = {n: self._feature(n) for n in feature_weights}
        signal = sig_mod.combine(feats, feature_weights)

        # Liquidity filter.
        signal = sig_mod.apply_liquidity_filter(
            signal, self._feature("dollar_vol_20d"), self.liquidity_threshold
        )

        # Membership mask.
        if self.membership_mask is not None:
            mask = self.membership_mask.reindex(
                index=signal.index, columns=signal.columns
            ).fillna(False)
            signal = signal.where(mask)

        # EWMA smoothing.
        if halflife_days and halflife_days > 0:
            signal = sig_mod.smooth_ewma(signal, halflife_days=halflife_days)

        # Sector + size neutralization (always on in research mode).
        size_df = self._feature("dollar_vol_20d")
        signal = neut_mod.neutralize(signal, sectors=self.sectors, size=size_df)

        prices = self.panel["adj_close"]
        res = run_backtest(
            signal,
            prices,
            n_deciles=self.n_deciles,
            cost_bps=self.cost_bps,
            weighting=weighting,
            exit_n_deciles=exit_n_deciles,
        )
        fwd_ret = prices.pct_change().shift(-1)
        ic = met.information_coefficient(signal, fwd_ret)
        summary = met.summary(res.returns, turnover=res.turnover)
        summary.update(met.ic_summary(ic))
        gross = met.summary(res.gross_returns)
        summary["gross_sharpe"] = gross["sharpe"]
        summary["gross_ann_return"] = gross["ann_return"]
        summary["cost_drag_ann"] = gross["ann_return"] - summary["ann_return"]
        config = {
            "feature_weights": feature_weights,
            "halflife_days": halflife_days,
            "weighting": weighting,
            "exit_n_deciles": exit_n_deciles,
        }
        summary["config"] = config

        run_id: str | None = None
        if self.journal is not None:
            run_id = self.journal.record_run(config, summary)
            summary["run_id"] = run_id

        # Keep the full artifacts so introspection tools can examine the run.
        artifact = RunArtifacts(
            run_id=run_id,
            signal=signal,
            fwd_ret=fwd_ret.reindex_like(signal),
            weights=res.weights,
            net_returns=res.returns,
            gross_returns=res.gross_returns,
            per_decile=res.per_decile_returns,
            summary=summary,
            config=config,
        )
        self.recent_runs.append(artifact)
        # Ring buffer.
        if len(self.recent_runs) > self.recent_runs_max:
            self.recent_runs = self.recent_runs[-self.recent_runs_max:]

        self.history.append(summary)
        return summary

    # ----- introspection --------------------------------------------------

    def last_run(self) -> "RunArtifacts | None":
        return self.recent_runs[-1] if self.recent_runs else None

    def analyze_last_run(self) -> dict:
        """Return per-decile spread + IC by year + cumulative equity for the last run."""
        art = self.last_run()
        if art is None:
            return {"error": "no runs recorded yet"}

        per_decile_ann = (art.per_decile.mean() * 252).round(4)
        per_decile_sharpe = (
            art.per_decile.mean() / art.per_decile.std().replace(0, np.nan) * np.sqrt(252)
        ).round(3)

        # IC by year.
        ic_series = met.information_coefficient(art.signal, art.fwd_ret)
        ic_by_year = {}
        for year, grp in ic_series.dropna().groupby(ic_series.dropna().index.year):
            mean = float(grp.mean())
            std = float(grp.std(ddof=0))
            ir = mean / std * np.sqrt(252) if std > 0 else float("nan")
            ic_by_year[int(year)] = {
                "ic_mean": round(mean, 4),
                "ic_ir": round(ir, 3),
                "n": int(grp.size),
            }

        # Equity curve: final value + max drawdown by year.
        eq = (1 + art.net_returns.fillna(0)).cumprod()
        eq_year_end = eq.resample("YE").last()

        return {
            "run_id": art.run_id,
            "config": art.config,
            "summary": {k: art.summary.get(k) for k in
                        ["sharpe", "gross_sharpe", "ic_mean", "ic_ir", "avg_turnover", "max_drawdown"]},
            "per_decile_annual_return": per_decile_ann.to_dict(),
            "per_decile_sharpe": per_decile_sharpe.to_dict(),
            "ic_by_year": ic_by_year,
            "equity_year_end": {str(k.year): round(float(v), 4) for k, v in eq_year_end.items()},
        }

    def feature_correlations(self, names: list[str]) -> dict:
        """Pearson correlations between raw feature values (flattened panel)."""
        # Need features computed.
        dfs: dict[str, pd.DataFrame] = {}
        for n in names:
            if n not in self.feature_fns:
                return {"error": f"unknown feature '{n}'"}
            dfs[n] = self._feature(n)
        # Flatten each to 1-D, aligning on (date, ticker).
        stacked = {n: df.stack(future_stack=True) for n, df in dfs.items()}
        combined = pd.concat(stacked, axis=1).dropna()
        if combined.empty:
            return {"error": "no overlapping non-NaN observations"}
        corr = combined.corr().round(3)
        return {
            "n_observations": int(len(combined)),
            "correlation_matrix": corr.to_dict(),
        }

    def feature_stats(self, name: str) -> dict:
        """Distribution + autocorrelation summary for a single feature."""
        if name not in self.feature_fns:
            return {"error": f"unknown feature '{name}'"}
        df = self._feature(name)
        flat = df.stack(future_stack=True).dropna()
        if flat.empty:
            return {"error": "feature is entirely NaN"}
        q = flat.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        nan_frac = 1 - len(flat) / (df.shape[0] * df.shape[1])
        # Temporal autocorrelation of the per-ticker rank — how fast does this signal rotate?
        ranks = df.rank(axis=1, pct=True)
        lag1_autocorr_per_ticker = ranks.apply(lambda s: s.autocorr(lag=1), axis=0)
        return {
            "feature": name,
            "n_obs": int(len(flat)),
            "nan_fraction": round(float(nan_frac), 4),
            "q01": round(float(q.loc[0.01]), 6),
            "q25": round(float(q.loc[0.25]), 6),
            "median": round(float(q.loc[0.5]), 6),
            "q75": round(float(q.loc[0.75]), 6),
            "q99": round(float(q.loc[0.99]), 6),
            "rank_autocorr_1d_mean": round(float(lag1_autocorr_per_ticker.mean()), 3),
            "rank_autocorr_1d_median": round(float(lag1_autocorr_per_ticker.median()), 3),
        }


# ----- Tool factory ---------------------------------------------------------


def build_tools(session: ResearchSession) -> list:
    """Return the @beta_tool-decorated callables bound to `session`."""

    @beta_tool
    def list_features() -> str:
        """List all registered feature functions available for backtesting.

        Returns a JSON array of feature names. The baseline features are
        `mom_12_1`, `reversal_5d`, `volume_shock`, `vol_21d`, `amihud_20d`,
        and `dollar_vol_20d` (used for liquidity filtering).
        """
        return json.dumps(sorted(session.feature_fns.keys()))

    @beta_tool
    def propose_feature(name: str, python_code: str, description: str) -> str:
        """Register a new price/volume feature function for backtesting.

        The code must define a single Python function named exactly `name` with
        signature `def <name>(panel: dict) -> pd.DataFrame`. The `panel` dict
        maps field names (`open`, `high`, `low`, `close`, `adj_close`, `volume`)
        to `pd.DataFrame` of shape (dates, tickers). The function must return a
        DataFrame of the same shape containing the feature values (higher =
        more long-favoured after normalization).

        Only `np` and `pd` are available (pre-injected). No imports, no file
        or network access, no eval/exec/getattr. Use rolling operations
        (`.rolling(N)`), pct_change, shift, etc. Handle NaN naturally — the
        combine step handles winsorization and z-scoring.

        Args:
            name: Feature identifier (snake_case, unique).
            python_code: Full source of the function definition.
            description: Short explanation of the economic intuition.

        Returns a status string. On error, the message explains what was
        rejected — read it and fix the code before retrying.
        """
        try:
            session.register_feature(name, python_code, description=description)
        except UnsafeCodeError as e:
            return f"REJECTED: {e}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: unexpected {type(e).__name__}: {e}"
        return f"OK: feature '{name}' registered ({description})"

    @beta_tool
    def run_backtest_tool(
        feature_weights: dict,
        halflife_days: int = 3,
        weighting: str = "decile_sticky",
        exit_n_deciles: int = 5,
    ) -> str:
        """Run a cross-sectional long/short backtest with the given settings.

        Pipeline: combine(weighted features) → liquidity filter → membership
        mask → EWMA smooth → sector+size neutralize → decile L/S backtest.

        Args:
            feature_weights: Map from feature name (must be registered) to
                weight. Raw weights are z-scored cross-sectionally before
                combination, so relative weights matter, not scale.
            halflife_days: EWMA halflife for signal smoothing (0 = off).
                Empirical sweet spot for this pipeline is 2-3d.
            weighting: One of "decile", "decile_sticky", "signal_weighted".
                `decile_sticky` with `exit_n_deciles=5` is the current best.
            exit_n_deciles: Only used for `decile_sticky`; wider exit band
                reduces churn (e.g. 5 = hold while in top 20%).

        Returns a JSON summary including gross_sharpe, sharpe (net),
        max_drawdown, avg_turnover, ic_mean, ic_ir, cost_drag_ann.
        Baseline (v4 stack from the grid) hits net sharpe ~+0.13.
        """
        try:
            summary = session.run(
                feature_weights=dict(feature_weights),
                halflife_days=halflife_days,
                weighting=weighting,
                exit_n_deciles=exit_n_deciles,
            )
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"
        return json.dumps(summary, indent=2, default=float)

    @beta_tool
    def analyze_last_run() -> str:
        """Diagnose the most recent backtest.

        Returns per-decile annualized returns + Sharpes, IC broken out by year
        (catches regime-dependent performance), and year-end cumulative equity.
        Use this AFTER a run_backtest_tool call — before proposing another
        feature — to understand WHY the run did what it did:
          * monotonic decile spread → signal rank-orders correctly
          * IC concentrated in 1-2 years → regime-dependent (fragile)
          * top decile + bottom decile both positive → signal not differentiating
          * equity climbs then collapses → structural break

        Returns JSON. No arguments (always analyzes the most recent run).
        """
        result = session.analyze_last_run()
        return json.dumps(result, indent=2, default=float)

    @beta_tool
    def feature_correlations(feature_names: list) -> str:
        """Pearson correlations between raw feature values (flattened across dates+tickers).

        Use this BEFORE proposing a feature to check whether your proposed
        variant is redundant with something already in the registry. A
        correlation above ~0.85 usually means the new feature is duplicating
        an existing one and won't add information after z-scoring.

        Args:
            feature_names: List of feature names to correlate. Include your
                proposed feature plus 3-5 existing baselines you suspect overlap.
        """
        result = session.feature_correlations(list(feature_names))
        return json.dumps(result, indent=2, default=float)

    @beta_tool
    def feature_stats(feature_name: str) -> str:
        """Distribution + rank-autocorrelation summary for one feature.

        Use this to sanity-check a proposed feature BEFORE running a full
        backtest. What to look for:
          * nan_fraction > 0.5 → feature sparse / broken on many names
          * q01 == q99 → degenerate (mostly a single value); won't differentiate
          * rank_autocorr_1d_mean < 0.3 → very noisy day-to-day; high turnover
          * rank_autocorr_1d_mean > 0.95 → too slow; essentially a level

        Args:
            feature_name: Feature to inspect.
        """
        result = session.feature_stats(feature_name)
        return json.dumps(result, indent=2, default=float)

    return [
        list_features,
        propose_feature,
        run_backtest_tool,
        analyze_last_run,
        feature_correlations,
        feature_stats,
    ]
