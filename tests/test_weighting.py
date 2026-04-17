from __future__ import annotations

import numpy as np
import pandas as pd

from quant_agent import signals as sig
from quant_agent.backtest import (
    signal_weighted_weights,
    sticky_decile_weights,
)


def test_smooth_ewma_constant_passthrough():
    idx = pd.bdate_range("2022-01-01", periods=30)
    df = pd.DataFrame(5.0, index=idx, columns=["A", "B"])
    out = sig.smooth_ewma(df, halflife_days=5)
    # EWMA of a constant is the constant (after initial row).
    assert np.allclose(out.values, 5.0)


def test_smooth_ewma_zero_halflife_identity():
    idx = pd.bdate_range("2022-01-01", periods=5)
    df = pd.DataFrame(np.arange(10).reshape(5, 2).astype(float), index=idx, columns=["A", "B"])
    out = sig.smooth_ewma(df, halflife_days=0)
    pd.testing.assert_frame_equal(out, df)


def test_smooth_ewma_reduces_day_to_day_change():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-01", periods=200)
    noisy = pd.DataFrame(rng.normal(size=(200, 5)), index=idx, columns=list("ABCDE"))
    smoothed = sig.smooth_ewma(noisy, halflife_days=5)
    # L1 daily change of smoothed signal must be smaller than raw.
    raw_churn = noisy.diff().abs().sum().sum()
    sm_churn = smoothed.diff().abs().sum().sum()
    assert sm_churn < raw_churn * 0.6  # substantial reduction


def test_signal_weighted_is_dollar_neutral_and_gross_two():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2022-01-01", periods=10)
    sig_df = pd.DataFrame(rng.normal(size=(10, 20)), index=idx, columns=[f"T{i}" for i in range(20)])
    w = signal_weighted_weights(sig_df, target_gross=2.0)
    # Per row: sum to ~0 (dollar neutral) and |w| sums to ~2.
    assert np.allclose(w.sum(axis=1).values, 0.0, atol=1e-10)
    assert np.allclose(w.abs().sum(axis=1).values, 2.0, atol=1e-10)


def test_signal_weighted_monotone_with_signal():
    """Highest signal → highest weight."""
    idx = pd.bdate_range("2022-01-01", periods=3)
    sig_df = pd.DataFrame(
        np.tile(np.arange(10.0), (3, 1)), index=idx, columns=[f"T{i}" for i in range(10)]
    )
    w = signal_weighted_weights(sig_df)
    # Weights should be monotonically increasing across columns.
    for i in range(3):
        row = w.iloc[i].values
        assert np.all(np.diff(row) > 0)
        # Lowest-signal names short, highest long.
        assert row[0] < 0 < row[-1]


def test_sticky_reduces_turnover_vs_hard_decile():
    """Construct noisy signal where a ticker hovers near the decile boundary.
    Sticky should hold it, hard decile should churn it."""
    from quant_agent.backtest import decile_weights

    rng = np.random.default_rng(42)
    n_dates, n_tickers = 300, 20
    idx = pd.bdate_range("2022-01-01", periods=n_dates)
    base = np.tile(np.arange(n_tickers, dtype=float), (n_dates, 1))
    noise = rng.normal(scale=2.0, size=base.shape)  # big noise near rank boundaries
    sig_df = pd.DataFrame(base + noise, index=idx, columns=[f"T{i}" for i in range(n_tickers)])

    w_hard = decile_weights(sig_df, n_deciles=10)
    w_sticky = sticky_decile_weights(sig_df, n_deciles=10, exit_n_deciles=5)

    tov_hard = w_hard.diff().abs().sum(axis=1).sum()
    tov_sticky = w_sticky.diff().abs().sum(axis=1).sum()
    assert tov_sticky < tov_hard


def test_sticky_state_machine_basic():
    """A ticker in the top decile stays long as long as it remains in the top exit band."""
    idx = pd.bdate_range("2022-01-01", periods=4)
    # 10 tickers. T9 in top decile day 0; drops to mid-rank day 1 (inside 20% band since only 10 names and 5 exit deciles gives top 2 spots); exits band day 2.
    # Simpler: design explicitly.
    cols = [f"T{i}" for i in range(10)]
    # Signal: T9 highest on day 0, day 1 = 8th (still in top 20% of 10 = 2 names: ranks 8,9), day 2 = 5th (out of top 20%), day 3 = 5th again.
    data = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # day 0: T9 top
            [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],  # day 1: T9 rank 8 (still in top 20%)
            [9, 1, 2, 3, 4, 5, 6, 7, 8, 0],  # day 2: T9 lowest (exits long)
            [9, 1, 2, 3, 4, 5, 6, 7, 8, 0],  # day 3: same
        ],
        dtype=float,
    )
    sig_df = pd.DataFrame(data, index=idx, columns=cols)
    w = sticky_decile_weights(sig_df, n_deciles=10, exit_n_deciles=5)
    # Day 0: T9 long.
    assert w.iloc[0]["T9"] > 0
    # Day 1: T9 held (rank 8 still in top 20% = top 2).
    assert w.iloc[1]["T9"] > 0
    # Day 2: T9 rank 0 — should be short now (it's the lowest), not long.
    assert w.iloc[2]["T9"] < 0
    # Day 3: still short.
    assert w.iloc[3]["T9"] < 0
