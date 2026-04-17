from __future__ import annotations

import numpy as np
import pandas as pd

from quant_agent.backtest import decile_weights, run_backtest


def test_decile_weights_dollar_neutral():
    idx = pd.date_range("2020-01-01", periods=5)
    # 10 tickers, signal = 0..9 identically each day.
    sig = pd.DataFrame(
        np.tile(np.arange(10, dtype=float), (5, 1)),
        index=idx,
        columns=[f"T{i}" for i in range(10)],
    )
    w = decile_weights(sig, n_deciles=10)
    # Longs = highest signal (T9), shorts = lowest (T0). Equal-weighted singletons.
    assert np.allclose(w["T9"].values, 1.0)
    assert np.allclose(w["T0"].values, -1.0)
    # Gross dollar exposure = 2 (1 long + 1 short).
    assert np.allclose(w.abs().sum(axis=1).values, 2.0)
    # Net dollar exposure = 0.
    assert np.allclose(w.sum(axis=1).values, 0.0)


def test_run_backtest_perfect_signal_positive():
    """Signal that perfectly predicts next-day return must produce positive PnL."""
    idx = pd.date_range("2020-01-01", periods=40)
    tickers = [f"T{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    # Construct next-day returns first, then prices consistent with them.
    fwd = pd.DataFrame(
        rng.normal(0, 0.01, size=(len(idx), len(tickers))),
        index=idx,
        columns=tickers,
    )
    # price[t+1]/price[t] - 1 = fwd[t]  =>  build price series by cumprod.
    gross_r = fwd.shift(1).fillna(0.0)  # daily return at t is fwd[t-1]
    prices = (1 + gross_r).cumprod() * 100.0
    # Perfect signal equals fwd_ret.
    res = run_backtest(fwd, prices, n_deciles=5, cost_bps=0.0)
    # Gross PnL should be strongly positive on average.
    assert res.gross_returns.dropna().mean() > 0


def test_cost_reduces_returns():
    idx = pd.date_range("2020-01-01", periods=30)
    tickers = [f"T{i}" for i in range(10)]
    rng = np.random.default_rng(1)
    sig = pd.DataFrame(
        rng.normal(size=(len(idx), len(tickers))), index=idx, columns=tickers
    )
    prices = (1 + rng.normal(0, 0.01, size=(len(idx), len(tickers)))).cumprod(axis=0) * 100
    prices = pd.DataFrame(prices, index=idx, columns=tickers)

    r0 = run_backtest(sig, prices, cost_bps=0.0, n_deciles=5)
    r5 = run_backtest(sig, prices, cost_bps=50.0, n_deciles=5)
    assert r5.returns.sum() < r0.returns.sum()
