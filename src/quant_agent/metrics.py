"""Performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ann_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float(r.mean() * TRADING_DAYS)


def ann_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series) -> float:
    v = ann_vol(returns)
    if v == 0 or np.isnan(v):
        return float("nan")
    return ann_return(returns) / v


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def summary(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    out = {
        "ann_return": ann_return(returns),
        "ann_vol": ann_vol(returns),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "skew": float(returns.dropna().skew()) if returns.dropna().size > 2 else float("nan"),
        "n_obs": int(returns.dropna().size),
    }
    if turnover is not None:
        out["avg_turnover"] = float(turnover.dropna().mean())
    return out


def decile_spread_table(per_decile: pd.DataFrame) -> pd.DataFrame:
    """Mean, Sharpe, and cumulative return per decile (annualized)."""
    rows = {}
    for d in per_decile.columns:
        s = per_decile[d].dropna()
        rows[d] = {
            "mean_ann": float(s.mean() * TRADING_DAYS) if not s.empty else float("nan"),
            "sharpe": float((s.mean() / s.std(ddof=0)) * np.sqrt(TRADING_DAYS))
            if not s.empty and s.std(ddof=0) > 0
            else float("nan"),
            "n": int(s.size),
        }
    return pd.DataFrame(rows).T.sort_index()


def information_coefficient(signal: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman rank correlation between signal and fwd return."""
    signal, fwd_returns = signal.align(fwd_returns, join="inner")

    def _row_ic(row_s: pd.Series, row_r: pd.Series) -> float:
        mask = row_s.notna() & row_r.notna()
        if mask.sum() < 10:
            return float("nan")
        a = row_s[mask].rank()
        b = row_r[mask].rank()
        return float(a.corr(b))

    ics = [
        _row_ic(signal.iloc[i], fwd_returns.iloc[i])
        for i in range(len(signal))
    ]
    return pd.Series(ics, index=signal.index, name="ic")


def ic_summary(ic: pd.Series) -> dict:
    s = ic.dropna()
    if s.empty:
        return {"ic_mean": float("nan"), "ic_ir": float("nan"), "n": 0}
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    ir = mean / std * np.sqrt(TRADING_DAYS) if std > 0 else float("nan")
    return {"ic_mean": mean, "ic_ir": ir, "n": int(s.size)}
