"""Shared test fixtures — synthetic price/volume panels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_panel() -> dict[str, pd.DataFrame]:
    """Deterministic 300-day panel for 8 tickers."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=300)
    tickers = [f"T{i}" for i in range(8)]
    rets = rng.normal(0.0005, 0.015, size=(len(dates), len(tickers)))
    # Inject cross-sectional structure: ticker i has drift proportional to i.
    rets += np.linspace(-0.001, 0.001, len(tickers))[None, :]
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    adj_close = pd.DataFrame(px, index=dates, columns=tickers)
    close = adj_close.copy()
    open_ = adj_close * (1 + rng.normal(0, 0.001, size=adj_close.shape))
    high = adj_close * (1 + np.abs(rng.normal(0, 0.005, size=adj_close.shape)))
    low = adj_close * (1 - np.abs(rng.normal(0, 0.005, size=adj_close.shape)))
    volume = pd.DataFrame(
        rng.integers(1_000_000, 10_000_000, size=adj_close.shape),
        index=dates,
        columns=tickers,
    ).astype(float)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": adj_close,
        "volume": volume,
    }
