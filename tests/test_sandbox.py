from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_agent.sandbox import UnsafeCodeError, exec_feature


def _panel(n_dates: int = 30, n_tickers: int = 5) -> dict:
    idx = pd.bdate_range("2022-01-01", periods=n_dates)
    cols = [f"T{i}" for i in range(n_tickers)]
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n_dates, n_tickers)), axis=0))
    df = pd.DataFrame(px, index=idx, columns=cols)
    vol = pd.DataFrame(
        rng.integers(1_000_000, 10_000_000, size=(n_dates, n_tickers)),
        index=idx,
        columns=cols,
    ).astype(float)
    return {
        "open": df,
        "high": df,
        "low": df,
        "close": df,
        "adj_close": df,
        "volume": vol,
    }


# ----- acceptance tests -----


def test_valid_feature_function_executes():
    src = (
        "def my_feat(panel):\n"
        "    px = panel['adj_close']\n"
        "    return px.pct_change(5)\n"
    )
    fn = exec_feature(src, expected_name="my_feat")
    out = fn(_panel())
    assert isinstance(out, pd.DataFrame)


def test_uses_np_and_pd():
    src = (
        "def feat(panel):\n"
        "    rets = panel['adj_close'].pct_change()\n"
        "    return np.log(rets.abs() + 1e-9).rolling(10).mean()\n"
    )
    fn = exec_feature(src, expected_name="feat")
    out = fn(_panel())
    assert out.shape == (30, 5)


def test_single_function_inferred_without_expected_name():
    src = "def only(panel): return panel['adj_close']"
    fn = exec_feature(src)
    assert callable(fn)


# ----- rejection tests -----


@pytest.mark.parametrize(
    "src,reason",
    [
        ("import os\ndef f(p): return p['adj_close']", "import"),
        ("from os import system\ndef f(p): return p['adj_close']", "import-from"),
        ("def f(p): return open('/etc/passwd').read()", "open"),
        ("def f(p): eval('1+1'); return p['adj_close']", "eval"),
        ("def f(p): exec('x=1'); return p['adj_close']", "exec"),
        ("def f(p): return __import__('os').system('ls')", "__import__"),
        ("def f(p): return p.__class__", "dunder access"),
        ("def f(p): return getattr(p, 'adj_close')", "getattr"),
        ("def f(p):\n    global x\n    return p['adj_close']", "global"),
    ],
)
def test_rejects_unsafe_source(src, reason):
    with pytest.raises(UnsafeCodeError):
        exec_feature(src)


def test_syntax_error_wrapped():
    with pytest.raises(UnsafeCodeError, match="syntax"):
        exec_feature("def f(p: return")


def test_missing_expected_function_name():
    src = "def other(panel): return panel['adj_close']"
    with pytest.raises(UnsafeCodeError):
        exec_feature(src, expected_name="wanted")


def test_multiple_functions_without_expected_name():
    src = (
        "def a(panel): return panel['adj_close']\n"
        "def b(panel): return panel['volume']\n"
    )
    with pytest.raises(UnsafeCodeError):
        exec_feature(src)
