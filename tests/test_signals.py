from __future__ import annotations

import numpy as np
import pandas as pd

from quant_agent import signals as sig


def test_zscore_row_mean_zero():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [4.0, 4.0, 4.0], "c": [7.0, 6.0, 5.0]}
    )
    z = sig.zscore(df, robust=False)
    # Each row should have mean ~0 across columns.
    assert np.allclose(z.mean(axis=1).values, 0.0)


def test_winsorize_clips():
    df = pd.DataFrame(
        [[-100.0, 0.0, 0.0, 0.0, 100.0]], columns=list("abcde")
    )
    w = sig.winsorize(df, pct=0.25)
    row = w.iloc[0]
    assert row.min() > -100.0
    assert row.max() < 100.0


def test_combine_respects_weights():
    idx = pd.date_range("2020-01-01", periods=3)
    f1 = pd.DataFrame([[1.0, -1.0]] * 3, index=idx, columns=["a", "b"])
    f2 = pd.DataFrame([[-1.0, 1.0]] * 3, index=idx, columns=["a", "b"])
    # Equal positive weights -> composite near zero.
    c = sig.combine({"f1": f1, "f2": f2}, {"f1": 1.0, "f2": 1.0})
    assert np.allclose(c.values, 0.0, atol=1e-10)
    # f1 alone -> a > b.
    c2 = sig.combine({"f1": f1, "f2": f2}, {"f1": 1.0, "f2": 0.0})
    assert (c2["a"] > c2["b"]).all()
