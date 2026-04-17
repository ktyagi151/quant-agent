from __future__ import annotations

import numpy as np
import pandas as pd

from quant_agent import neutralize as neut


def _make_signal(n_dates=50, n_tickers=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=n_dates)
    cols = [f"T{i}" for i in range(n_tickers)]
    return pd.DataFrame(rng.normal(size=(n_dates, n_tickers)), index=idx, columns=cols)


def test_sector_neutralize_row_means_zero_per_sector():
    signal = _make_signal()
    sectors = pd.Series(
        {t: ("A" if i < 20 else "B") for i, t in enumerate(signal.columns)}
    )
    out = neut.sector_neutralize(signal, sectors)
    a_cols = [t for t in signal.columns if sectors[t] == "A"]
    b_cols = [t for t in signal.columns if sectors[t] == "B"]
    assert np.allclose(out[a_cols].mean(axis=1).values, 0.0, atol=1e-10)
    assert np.allclose(out[b_cols].mean(axis=1).values, 0.0, atol=1e-10)


def test_size_neutralize_zero_correlation_with_log_size():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2022-01-01", periods=30)
    cols = [f"T{i}" for i in range(25)]
    # Make signal strongly correlated with size so the residual must de-correlate.
    size = pd.DataFrame(
        rng.uniform(1e6, 1e9, size=(len(idx), len(cols))),
        index=idx,
        columns=cols,
    )
    log_size = np.log(size)
    signal = 2.0 * log_size + pd.DataFrame(
        rng.normal(scale=0.1, size=log_size.shape), index=idx, columns=cols
    )
    resid = neut.size_neutralize(signal, size)
    # Row-wise correlations between residual and log_size should be near zero.
    corrs = []
    for i in range(len(idx)):
        r = resid.iloc[i].corr(log_size.iloc[i])
        corrs.append(r)
    assert np.nanmax(np.abs(corrs)) < 1e-8


def test_combined_neutralize_both_regressors():
    rng = np.random.default_rng(7)
    signal = _make_signal(seed=7)
    # Inject a sector effect and a size effect that should get removed.
    sectors = pd.Series(
        {t: ("X" if i % 2 == 0 else "Y") for i, t in enumerate(signal.columns)}
    )
    size = pd.DataFrame(
        rng.uniform(1e7, 1e9, size=signal.shape),
        index=signal.index,
        columns=signal.columns,
    )
    log_size = np.log(size)
    sector_effect = pd.Series(
        {t: (1.0 if sectors[t] == "X" else -1.0) for t in signal.columns}
    )
    biased = signal + sector_effect + 0.5 * log_size

    out = neut.neutralize(biased, sectors=sectors, size=size)

    # Each sector mean per row should be ~zero.
    x_cols = [t for t in biased.columns if sectors[t] == "X"]
    assert np.nanmax(np.abs(out[x_cols].mean(axis=1).values)) < 1e-8
    # Residual must have near-zero correlation with log_size per row.
    for i in range(len(out)):
        c = out.iloc[i].corr(log_size.iloc[i])
        assert abs(c) < 1e-6


def test_handles_nan_rows_gracefully():
    signal = _make_signal()
    # Introduce NaNs.
    signal.iloc[0, :35] = np.nan  # too few valid -> whole row stays NaN
    sectors = pd.Series(
        {t: ("A" if i < 20 else "B") for i, t in enumerate(signal.columns)}
    )
    out = neut.neutralize(signal, sectors=sectors, size=None)
    assert out.iloc[0].isna().all()
    assert out.iloc[1].notna().sum() > 0
