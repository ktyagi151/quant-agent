"""Vectorized cross-sectional decile long/short backtest."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series            # daily net strategy return (after costs)
    gross_returns: pd.Series      # before costs
    weights: pd.DataFrame         # date x ticker portfolio weights (built from signal[t])
    turnover: pd.Series           # per-day turnover (sum |Δw| / 2)
    per_decile_returns: pd.DataFrame  # date x decile, gross fwd return by bucket
    cost_bps: float


def _assign_deciles(row: pd.Series, n: int) -> pd.Series:
    valid = row.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=row.index)
    try:
        q = pd.qcut(valid.rank(method="first"), n, labels=False)
    except ValueError:
        return pd.Series(np.nan, index=row.index)
    out = pd.Series(np.nan, index=row.index)
    out.loc[valid.index] = q.values
    return out


def decile_weights(signal: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """Build dollar-neutral equal-weighted decile L/S weights from a signal.

    Sum of long weights = +1, sum of short weights = -1, gross = 2.
    """
    deciles = signal.apply(_assign_deciles, axis=1, args=(n_deciles,))
    top = (deciles == n_deciles - 1).astype(float)
    bot = (deciles == 0).astype(float)
    n_top = top.sum(axis=1).replace(0, np.nan)
    n_bot = bot.sum(axis=1).replace(0, np.nan)
    long_w = top.div(n_top, axis=0)
    short_w = bot.div(n_bot, axis=0)
    return (long_w - short_w).fillna(0.0)


def sticky_decile_weights(
    signal: pd.DataFrame,
    n_deciles: int = 10,
    exit_n_deciles: int | None = None,
) -> pd.DataFrame:
    """Hysteresis-banded decile L/S.

    A ticker ENTERS the long side when its rank is in the top 1/n_deciles
    (e.g. top 10%). It STAYS long while its rank remains in the top
    1/exit_n_deciles (e.g. top 20%). Symmetric short side.

    Path-dependent so implemented with a per-date loop (cheap for daily data).
    """
    if exit_n_deciles is None:
        exit_n_deciles = max(1, n_deciles // 2)
    if exit_n_deciles > n_deciles:
        raise ValueError("exit_n_deciles must be <= n_deciles (wider exit band)")

    entry = signal.apply(_assign_deciles, axis=1, args=(n_deciles,)).to_numpy()
    exit_ = signal.apply(_assign_deciles, axis=1, args=(exit_n_deciles,)).to_numpy()

    n_rows, n_cols = signal.shape
    long_state = np.zeros((n_rows, n_cols), dtype=bool)
    short_state = np.zeros((n_rows, n_cols), dtype=bool)
    prev_long = np.zeros(n_cols, dtype=bool)
    prev_short = np.zeros(n_cols, dtype=bool)

    for i in range(n_rows):
        top_enter = entry[i] == (n_deciles - 1)
        bot_enter = entry[i] == 0
        top_hold = exit_[i] == (exit_n_deciles - 1)
        bot_hold = exit_[i] == 0
        # NaN decile rows produce all-False entry/hold, so state naturally
        # clears for tickers missing that day.
        long_state[i] = top_enter | (prev_long & top_hold)
        short_state[i] = bot_enter | (prev_short & bot_hold)
        # A ticker can't be both; resolve ties in favour of the entry signal.
        conflict = long_state[i] & short_state[i]
        long_state[i] &= ~conflict | top_enter
        short_state[i] &= ~conflict | bot_enter
        prev_long = long_state[i]
        prev_short = short_state[i]

    long_w = long_state.astype(float)
    short_w = short_state.astype(float)
    n_long = long_w.sum(axis=1, keepdims=True)
    n_short = short_w.sum(axis=1, keepdims=True)
    long_w = np.divide(long_w, n_long, where=n_long > 0, out=np.zeros_like(long_w))
    short_w = np.divide(short_w, n_short, where=n_short > 0, out=np.zeros_like(short_w))
    w = long_w - short_w
    return pd.DataFrame(w, index=signal.index, columns=signal.columns)


def signal_weighted_weights(signal: pd.DataFrame, target_gross: float = 2.0) -> pd.DataFrame:
    """Dollar-neutral, signal-proportional weights.

    Per row:
      1. Demean signal across names → dollar-neutral.
      2. Rescale so sum |w| = target_gross (matches the decile scheme's gross = 2).
    """
    if signal.empty:
        return signal
    demeaned = signal.sub(signal.mean(axis=1), axis=0)
    l1 = demeaned.abs().sum(axis=1).replace(0, np.nan)
    w = demeaned.div(l1, axis=0) * target_gross
    return w.fillna(0.0)


def per_decile_returns(
    signal: pd.DataFrame, fwd_returns: pd.DataFrame, n_deciles: int = 10
) -> pd.DataFrame:
    """Equal-weighted gross forward return per decile per day."""
    deciles = signal.apply(_assign_deciles, axis=1, args=(n_deciles,))
    fr = fwd_returns.reindex_like(deciles)
    cols = {}
    for d in range(n_deciles):
        mask = (deciles == d).astype(float)
        n = mask.sum(axis=1).replace(0, np.nan)
        cols[d] = (mask * fr).sum(axis=1) / n
    return pd.DataFrame(cols)


WEIGHTING_SCHEMES = ("decile", "decile_sticky", "signal_weighted")


def run_backtest(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    n_deciles: int = 10,
    cost_bps: float = 5.0,
    weighting: str = "decile",
    exit_n_deciles: int | None = None,
) -> BacktestResult:
    """Vectorized daily-rebalanced L/S backtest.

    Convention: signal[t] is observed at close t; weights w[t] are rebalanced at
    close t; return earned on day t+1 is fwd_ret[t] = price[t+1]/price[t] - 1.
    Strategy return attributed to date t: w[t] * fwd_ret[t] minus turnover cost.

    weighting:
      * "decile"          — equal-weight top/bottom decile (baseline).
      * "decile_sticky"   — enter top decile, hold while in top `exit_n_deciles` band.
      * "signal_weighted" — weights proportional to demeaned signal, gross = 2.
    """
    if weighting not in WEIGHTING_SCHEMES:
        raise ValueError(f"weighting must be one of {WEIGHTING_SCHEMES}, got {weighting!r}")

    fwd_ret = prices.pct_change().shift(-1).reindex_like(signal)

    if weighting == "decile":
        w = decile_weights(signal, n_deciles=n_deciles)
    elif weighting == "decile_sticky":
        w = sticky_decile_weights(signal, n_deciles=n_deciles, exit_n_deciles=exit_n_deciles)
    else:  # signal_weighted
        w = signal_weighted_weights(signal, target_gross=2.0)

    gross = (w * fwd_ret).sum(axis=1)

    # Turnover: half of L1 change in weights (buys side only).
    dw = w.diff().abs().sum(axis=1).div(2.0)
    dw.iloc[0] = w.iloc[0].abs().sum() / 2.0

    cost = dw * (cost_bps / 1e4)
    net = gross - cost

    pdr = per_decile_returns(signal, fwd_ret, n_deciles=n_deciles)

    return BacktestResult(
        returns=net,
        gross_returns=gross,
        weights=w,
        turnover=dw,
        per_decile_returns=pdr,
        cost_bps=cost_bps,
    )
