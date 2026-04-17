"""Cross-sectional neutralization (sector, size).

Sector-demean is equivalent to OLS residualization on sector dummies (with an
intercept). We therefore implement one unified routine `residualize` that
takes arbitrary regressor DataFrames / ticker-attribute Series and returns
per-date OLS residuals of the signal on `[intercept, ...regressors...]`.

Size proxy: we use `log(dollar_vol_20d)` since yfinance does not give a clean
point-in-time market cap. This is an approximation — dollar volume correlates
with market cap but imperfectly. Noted in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_ROW_OBS = 10  # skip rows with too few usable names


def _build_sector_dummies(sectors: pd.Series, tickers: list[str]) -> pd.DataFrame:
    """Return ticker × (n_sectors - 1) float matrix. One sector dropped for identifiability.

    Sector values for tickers not in `sectors` are filled with 'UNKNOWN' so
    the design matrix stays full-rank.
    """
    s = sectors.reindex(tickers).fillna("UNKNOWN").astype(str)
    dummies = pd.get_dummies(s, drop_first=True).astype(float)
    dummies.index = tickers
    return dummies


def sector_neutralize(signal: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Per-row demean within sector. Vectorized via groupby."""
    if signal.empty:
        return signal
    tickers = list(signal.columns)
    sect = sectors.reindex(tickers).fillna("UNKNOWN").astype(str)
    result = signal.copy()
    for name, group in sect.groupby(sect):
        cols = group.index.tolist()
        block = signal[cols]
        means = block.mean(axis=1)
        result[cols] = block.sub(means, axis=0)
    return result


def size_neutralize(signal: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame:
    """Per-row OLS residual of signal on log(size). Both shaped [date × ticker]."""
    if signal.empty or size.empty:
        return signal
    return _residualize_per_row(signal, {"log_size": np.log(size.replace({0: np.nan}))})


def neutralize(
    signal: pd.DataFrame,
    sectors: pd.Series | None = None,
    size: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combined sector + size neutralization via a single per-row OLS.

    Regressors:
      * intercept
      * k-1 sector dummies (if `sectors` given)
      * log(size) (if `size` given)
    """
    if sectors is None and size is None:
        return signal
    if signal.empty:
        return signal

    tickers = list(signal.columns)
    ticker_regressors: dict[str, pd.Series] = {}
    daily_regressors: dict[str, pd.DataFrame] = {}

    if sectors is not None:
        dummies = _build_sector_dummies(sectors, tickers)
        for col in dummies.columns:
            ticker_regressors[f"sec_{col}"] = dummies[col]

    if size is not None:
        daily_regressors["log_size"] = np.log(size.replace({0: np.nan}))

    return _residualize_per_row(signal, daily_regressors, ticker_regressors)


def _residualize_per_row(
    signal: pd.DataFrame,
    daily_regressors: dict[str, pd.DataFrame] | None = None,
    ticker_regressors: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Per-row OLS residuals of signal on [intercept, daily..., ticker...].

    daily_regressors are date×ticker (e.g. log_size).
    ticker_regressors are ticker-attribute (broadcast across dates; e.g. sector dummies).
    """
    daily_regressors = daily_regressors or {}
    ticker_regressors = ticker_regressors or {}

    tickers = list(signal.columns)
    # Pre-materialize aligned numpy arrays.
    y_mat = signal.to_numpy(dtype=float)
    daily_arr = {
        name: df.reindex(index=signal.index, columns=tickers).to_numpy(dtype=float)
        for name, df in daily_regressors.items()
    }
    ticker_arr = {
        name: s.reindex(tickers).to_numpy(dtype=float)
        for name, s in ticker_regressors.items()
    }

    out = np.full_like(y_mat, np.nan)

    for i in range(y_mat.shape[0]):
        y_row = y_mat[i]
        regressor_cols = [np.ones_like(y_row)]  # intercept
        for _, arr in daily_arr.items():
            regressor_cols.append(arr[i])
        for _, vec in ticker_arr.items():
            regressor_cols.append(vec)

        X = np.column_stack(regressor_cols)
        mask = np.isfinite(y_row) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < _MIN_ROW_OBS:
            continue

        X_m = X[mask]
        y_m = y_row[mask]
        try:
            beta, *_ = np.linalg.lstsq(X_m, y_m, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = y_m - X_m @ beta
        out[i, mask] = resid

    return pd.DataFrame(out, index=signal.index, columns=signal.columns)
