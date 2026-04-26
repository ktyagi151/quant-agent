"""Portfolio construction tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_agent.optimization import (
    mean_variance_weights,
    risk_parity_weights,
    shrinkage_covariance,
    signal_weighted_constrained,
)


@pytest.fixture
def small_returns():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2022-01-01", periods=300)
    cols = [f"T{i}" for i in range(10)]
    return pd.DataFrame(rng.normal(0, 0.01, size=(300, 10)), index=idx, columns=cols)


@pytest.fixture
def small_signal(small_returns):
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.normal(size=small_returns.shape),
        index=small_returns.index,
        columns=small_returns.columns,
    )


def test_shrinkage_cov_diagonal_dominant():
    # Pure shrinkage=1 should be exactly diagonal.
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(size=(252, 5)), columns=list("ABCDE"))
    cov = shrinkage_covariance(rets, shrinkage=1.0)
    off_diag = cov.values - np.diag(np.diag(cov.values))
    assert np.allclose(off_diag, 0)


def test_signal_weighted_constrained_dollar_neutral(small_signal):
    # 10 names × cap=0.30 → max feasible gross = 10×0.30 = 3.0, target 2.0 fits
    w = signal_weighted_constrained(small_signal, target_gross=2.0, max_single_name=0.30)
    # Each row should sum to ~0 (dollar-neutral)
    assert np.allclose(w.sum(axis=1).values, 0, atol=1e-5)
    # Gross exposure should be near target (within rounding)
    avg_gross = float(w.abs().sum(axis=1).mean())
    assert 1.5 < avg_gross <= 2.0 + 1e-9


def test_signal_weighted_respects_single_name_cap(small_signal):
    # Cap is hard: |w| must respect it even if gross can't reach target
    w = signal_weighted_constrained(small_signal, target_gross=2.0, max_single_name=0.10)
    assert (w.abs() <= 0.10 + 1e-9).all().all()
    # Dollar-neutrality must still hold
    assert np.allclose(w.sum(axis=1).values, 0, atol=1e-5)


def test_signal_weighted_respects_sector_cap(small_signal):
    sectors = pd.Series(
        {col: ("A" if i < 5 else "B") for i, col in enumerate(small_signal.columns)}
    )
    w = signal_weighted_constrained(
        small_signal, sectors=sectors, target_gross=2.0,
        max_single_name=0.5, max_sector_weight=0.6,
    )
    sect_exposure = w.abs().T.groupby(sectors.reindex(w.columns)).sum().T
    # Each sector capped at 0.6 + slack from rebalancing
    assert (sect_exposure <= 0.65).all().all()


def test_mean_variance_returns_correct_shape(small_signal, small_returns):
    w = mean_variance_weights(small_signal, small_returns, target_te=0.10, rebalance_every=21)
    assert w.shape == small_signal.shape


def test_risk_parity_positive_weights(small_returns):
    w = risk_parity_weights(small_returns, lookback=100, rebalance_every=21)
    nonzero = w[w.abs().sum(axis=1) > 0]
    if not nonzero.empty:
        # Risk parity is long-only by construction
        assert (nonzero >= 0).all().all()
