"""Price/volume feature library.

Each feature takes a wide panel dict ({field: DataFrame[date x ticker]})
and returns a DataFrame[date x ticker] of raw (pre-normalization) values.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


FeatureFn = Callable[[dict[str, pd.DataFrame]], pd.DataFrame]


def mom_12_1(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """12-month return skipping the most recent month (classic momentum)."""
    px = panel["adj_close"]
    return px.shift(21) / px.shift(252) - 1.0


def reversal_5d(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Short-term reversal: negative of the last 5d return."""
    px = panel["adj_close"]
    return -px.pct_change(5)


def vol_21d(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """21-day realized vol of daily returns."""
    rets = panel["adj_close"].pct_change()
    return rets.rolling(21).std()


def volume_shock(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """log(volume / 20d average volume)."""
    v = panel["volume"].replace(0, np.nan)
    avg = v.rolling(20).mean()
    return np.log(v / avg)


def dollar_vol_20d(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """20d average dollar volume (liquidity filter input)."""
    return (panel["close"] * panel["volume"]).rolling(20).mean()


def amihud_20d(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Amihud illiquidity: rolling mean of |ret| / dollar_volume over 20d."""
    rets = panel["adj_close"].pct_change().abs()
    dv = (panel["close"] * panel["volume"]).replace(0, np.nan)
    return (rets / dv).rolling(20).mean()


# Registry so an LLM layer can introspect / extend later.
FEATURES: dict[str, FeatureFn] = {
    "mom_12_1": mom_12_1,
    "reversal_5d": reversal_5d,
    "vol_21d": vol_21d,
    "volume_shock": volume_shock,
    "dollar_vol_20d": dollar_vol_20d,
    "amihud_20d": amihud_20d,
}


def compute_features(panel: dict[str, pd.DataFrame], names: list[str]) -> dict[str, pd.DataFrame]:
    """Compute the requested features from the registry."""
    out = {}
    for n in names:
        if n not in FEATURES:
            raise KeyError(f"unknown feature: {n}. known: {sorted(FEATURES)}")
        out[n] = FEATURES[n](panel)
    return out
