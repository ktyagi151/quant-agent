"""Portfolio construction methods beyond the existing decile L/S.

The current `backtest.py` has three weighting schemes (`decile`,
`decile_sticky`, `signal_weighted`). This module adds the standard
quant-PM toolkit:

  * `mean_variance_weights` — Markowitz, with target tracking error
  * `risk_parity_weights` — equal risk contribution (no signal needed)
  * `signal_weighted_constrained` — like the existing scheme but with sector,
     single-name, gross, and net exposure caps

All methods take a `signal: DataFrame[date × ticker]` and a covariance source
(or use a shrinkage estimator), respect IPS hard constraints when given an
IPS, and return a `weights: DataFrame[date × ticker]`.

For the agent layer, the portfolio-construction agent picks among these
methods and tunes their parameters per the IPS soft constraints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ips import IPS, HardConstraint


# ----- shrinkage covariance --------------------------------------------------


def shrinkage_covariance(
    returns: pd.DataFrame, lookback: int = 252, shrinkage: float = 0.5
) -> pd.DataFrame:
    """Ledoit-Wolf style shrinkage of sample covariance toward diagonal.

    Returns the covariance matrix as of the *latest* date in `returns`, computed
    from the trailing `lookback` rows.
    """
    if len(returns) < lookback:
        sample = returns.dropna(how="all").cov()
    else:
        sample = returns.tail(lookback).cov()
    if sample.empty:
        return sample
    diag = np.diag(np.diag(sample.values))
    diag_df = pd.DataFrame(diag, index=sample.index, columns=sample.columns)
    return shrinkage * diag_df + (1 - shrinkage) * sample


# ----- mean-variance ---------------------------------------------------------


def mean_variance_weights(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    target_te: float = 0.10,
    lookback: int = 252,
    rebalance_every: int = 21,
    long_only: bool = False,
) -> pd.DataFrame:
    """Mean-variance weights with a tracking-error target.

    For each rebalance date, solve the unconstrained optimum
        w ∝ Σ⁻¹ μ
    where μ is the cross-sectional signal at that date and Σ is the shrunk
    covariance from the trailing `lookback` window. Then scale w to hit the
    target TE (annualized). Between rebalances, weights stay flat (drift
    ignored — this is a daily-rebalance frame).

    `long_only=True` zeros out negative weights post-scaling.

    Returns a weights DataFrame with the same shape as `signal`.
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    rebalance_dates = signal.index[::rebalance_every]
    last_w: pd.Series | None = None

    for d in signal.index:
        if d in rebalance_dates:
            mu = signal.loc[d].dropna()
            if len(mu) < 5:
                if last_w is not None:
                    weights.loc[d] = last_w
                continue
            window = returns.loc[:d, mu.index].tail(lookback)
            cov = shrinkage_covariance(window)
            cov = cov.reindex(index=mu.index, columns=mu.index).fillna(0.0)
            try:
                inv = np.linalg.pinv(cov.values + 1e-8 * np.eye(len(mu)))
            except np.linalg.LinAlgError:
                if last_w is not None:
                    weights.loc[d] = last_w
                continue
            raw = inv @ mu.values
            if long_only:
                raw = np.clip(raw, 0, None)
            # Scale to hit target TE: TE = sqrt(w' Σ w) annualized
            te_unscaled = float(np.sqrt(max(raw @ cov.values @ raw, 1e-12)) * np.sqrt(252))
            scale = target_te / te_unscaled if te_unscaled > 1e-12 else 1.0
            w = pd.Series(raw * scale, index=mu.index)
            full = pd.Series(0.0, index=signal.columns)
            full.loc[w.index] = w
            last_w = full
            weights.loc[d] = full
        elif last_w is not None:
            weights.loc[d] = last_w

    return weights


# ----- risk parity -----------------------------------------------------------


def risk_parity_weights(
    returns: pd.DataFrame,
    lookback: int = 252,
    rebalance_every: int = 21,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Equal-risk-contribution (ERC) weights, no alpha signal needed.

    Each name contributes equally to total portfolio variance. This is the
    "diversification" baseline — useful as a comparison or as the long leg of
    a long-only sleeve. Negative weights are not allowed (not well-defined
    for ERC).

    Algorithm: cyclical coordinate descent on the ERC condition (standard,
    converges in <200 iters for ≤500 names).
    """
    dates = returns.index
    cols = returns.columns
    weights = pd.DataFrame(0.0, index=dates, columns=cols)
    rebalance_dates = dates[::rebalance_every]
    last_w: pd.Series | None = None

    for d in dates:
        if d in rebalance_dates:
            window = returns.loc[:d].tail(lookback).dropna(axis=1, how="any")
            if window.shape[0] < 30 or window.shape[1] < 5:
                if last_w is not None:
                    weights.loc[d] = last_w
                continue
            cov = window.cov().values
            n = cov.shape[0]
            w = np.full(n, 1.0 / n)
            for _ in range(max_iter):
                marginal = cov @ w
                rc = w * marginal
                target = rc.mean()
                grad = rc - target
                step = grad / (marginal + 1e-12)
                w_new = np.clip(w - 0.1 * step, 1e-6, None)
                w_new /= w_new.sum()
                if np.max(np.abs(w_new - w)) < tol:
                    w = w_new
                    break
                w = w_new
            full = pd.Series(0.0, index=cols)
            full.loc[window.columns] = w
            last_w = full
            weights.loc[d] = full
        elif last_w is not None:
            weights.loc[d] = last_w

    return weights


# ----- constrained signal-weighted ------------------------------------------


def signal_weighted_constrained(
    signal: pd.DataFrame,
    sectors: pd.Series | None = None,
    target_gross: float = 2.0,
    max_single_name: float = 0.05,
    max_sector_weight: float = 0.30,
    long_only: bool = False,
) -> pd.DataFrame:
    """Signal-weighted, dollar-neutral, with per-name and sector caps.

    Per row:
      1. Demean signal cross-sectionally → dollar-neutral baseline.
      2. Rescale to target gross exposure.
      3. Cap each name at `max_single_name` (absolute).
      4. If `sectors` provided, cap aggregate sector exposure at `max_sector_weight`.

    Capping is iterative — after capping, gross may fall below target, so
    redistribute to uncapped names; up to 5 passes.
    """
    if signal.empty:
        return signal

    out = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    sect_map = sectors.reindex(signal.columns).fillna("UNKNOWN") if sectors is not None else None

    for d, row in signal.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        if long_only:
            v = np.clip(valid.values - valid.values.min(), 0, None)
        else:
            v = valid.values - valid.values.mean()
        l1 = np.sum(np.abs(v))
        if l1 <= 0:
            continue
        w = v / l1 * target_gross
        w_series = pd.Series(w, index=valid.index)

        # Single-name cap: clip + re-demean to fixpoint. Scale DOWN (not up)
        # to target gross — caps are hard, gross is soft (it can be below target
        # if the cap forces it).
        for _ in range(15):
            clipped = w_series.clip(lower=-max_single_name, upper=max_single_name)
            if long_only:
                next_w = clipped
            else:
                # Re-demean. May push some past cap again; loop.
                next_w = clipped - clipped.mean()
            if np.allclose(w_series.values, next_w.values, atol=1e-10):
                w_series = next_w
                break
            w_series = next_w
        # Final hard clip — cap is non-negotiable
        w_series = w_series.clip(lower=-max_single_name, upper=max_single_name)
        if not long_only:
            w_series = w_series - w_series.mean()
            # One more clip in case demean pushed an edge case over
            w_series = w_series.clip(lower=-max_single_name, upper=max_single_name)
        # Scale DOWN only (never up — would violate cap)
        gross_now = w_series.abs().sum()
        if gross_now > target_gross:
            w_series = w_series * (target_gross / gross_now)

        # Sector cap (after single-name cap is settled)
        if sect_map is not None:
            for _ in range(10):
                w_by_sect = w_series.abs().groupby(sect_map.reindex(w_series.index)).sum()
                violators = w_by_sect[w_by_sect > max_sector_weight + 1e-9]
                if violators.empty:
                    break
                for sect, total in violators.items():
                    in_sect = sect_map.reindex(w_series.index) == sect
                    scale = max_sector_weight / total
                    w_series.loc[in_sect] *= scale
                # Re-demean after sector scaling to restore neutrality
                if not long_only:
                    w_series = w_series - w_series.mean()

        full = pd.Series(0.0, index=signal.columns)
        full.loc[w_series.index] = w_series
        out.loc[d] = full

    return out


# ----- IPS helper ------------------------------------------------------------


def hard_constraint_threshold(ips: IPS, metric: str) -> float | None:
    """Pull a single hard-constraint threshold from the IPS, or None."""
    for hc in ips.hard_constraints:
        if hc.metric == metric:
            return hc.threshold
    return None


def construct_from_ips(
    method: str,
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    ips: IPS,
    sectors: pd.Series | None = None,
    target_te: float = 0.10,
) -> pd.DataFrame:
    """Dispatch helper: run the chosen method respecting the IPS caps."""
    max_name = hard_constraint_threshold(ips, "max_single_name_weight") or 0.05
    max_sector = hard_constraint_threshold(ips, "max_sector_weight") or 0.30
    gross_cap = hard_constraint_threshold(ips, "gross_exposure") or 2.0

    if method == "signal_weighted_constrained":
        return signal_weighted_constrained(
            signal=signal,
            sectors=sectors,
            target_gross=gross_cap,
            max_single_name=max_name,
            max_sector_weight=max_sector,
        )
    if method == "mean_variance":
        return mean_variance_weights(signal=signal, returns=returns, target_te=target_te)
    if method == "risk_parity":
        return risk_parity_weights(returns=returns)
    raise ValueError(
        f"unknown method {method!r}; supported: signal_weighted_constrained | mean_variance | risk_parity"
    )
