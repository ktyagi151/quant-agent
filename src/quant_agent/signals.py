"""Cross-sectional normalization and composite signal construction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(df: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
    """Cross-sectional (per-row) winsorization at [pct, 1-pct] quantiles."""
    if df.empty:
        return df
    lo = df.quantile(pct, axis=1)
    hi = df.quantile(1 - pct, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore(df: pd.DataFrame, robust: bool = True) -> pd.DataFrame:
    """Cross-sectional z-score per row.

    robust=True uses median + 1.4826*MAD; otherwise mean + std.
    """
    if df.empty:
        return df
    if robust:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1)
        scale = 1.4826 * mad.replace(0, np.nan)
        return df.sub(med, axis=0).div(scale, axis=0)
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def combine(features: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Weighted sum of per-row-zscored features. Handles missing keys."""
    if not features:
        return pd.DataFrame()
    aligned = None
    total_w = 0.0
    for name, w in weights.items():
        if name not in features:
            continue
        z = zscore(winsorize(features[name]))
        total_w += abs(w)
        aligned = z * w if aligned is None else aligned.add(z * w, fill_value=0.0)
    if aligned is None:
        return pd.DataFrame()
    return aligned / (total_w if total_w > 0 else 1.0)


def smooth_ewma(signal: pd.DataFrame, halflife_days: float) -> pd.DataFrame:
    """Exponentially-weighted moving average per ticker along the time axis.

    halflife_days <= 0 returns the input unchanged. Used to dampen day-to-day
    signal churn and reduce turnover before the weighting step.
    """
    if signal.empty or halflife_days is None or halflife_days <= 0:
        return signal
    return signal.ewm(halflife=halflife_days, adjust=False, ignore_na=True).mean()


def apply_liquidity_filter(
    signal: pd.DataFrame, dollar_vol: pd.DataFrame, min_usd: float
) -> pd.DataFrame:
    """NaN-out positions where 20d dollar volume is below the threshold."""
    if signal.empty or dollar_vol.empty:
        return signal
    mask = dollar_vol.reindex_like(signal) >= min_usd
    return signal.where(mask)
