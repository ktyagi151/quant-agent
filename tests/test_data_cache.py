"""Tests for the data-cache path logic (no network)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_agent import data as data_mod
from quant_agent import io_utils


@pytest.fixture
def tmp_cache(monkeypatch, tmp_path: Path):
    """Redirect raw_dir/meta_dir to a tmp path."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "meta").mkdir()
    monkeypatch.setattr(io_utils, "raw_dir", lambda: tmp_path / "raw")
    monkeypatch.setattr(io_utils, "meta_dir", lambda: tmp_path / "meta")
    # ticker_path closes over raw_dir — patch it too.
    monkeypatch.setattr(io_utils, "ticker_path", lambda t: tmp_path / "raw" / f"{t}.parquet")
    monkeypatch.setattr(data_mod, "ticker_path", lambda t: tmp_path / "raw" / f"{t}.parquet")
    return tmp_path


def _make_cached_frame(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adj_close": 1.0,
            "volume": 100.0,
        },
        index=idx,
    )


def test_load_panel_reads_cache_only(tmp_cache: Path):
    t = "AAA"
    idx = pd.bdate_range("2024-01-01", periods=10)
    df = _make_cached_frame(idx)
    df.to_parquet(tmp_cache / "raw" / f"{t}.parquet")
    out = data_mod.load_panel([t], "2024-01-01", "2024-01-15")
    assert t in out
    assert len(out[t]) == 10


def test_load_panel_missing_ticker_empty(tmp_cache: Path):
    out = data_mod.load_panel(["MISSING"], "2024-01-01", "2024-01-15")
    assert "MISSING" in out
    assert out["MISSING"].empty


def test_to_wide_panel_shape(tmp_cache: Path):
    idx = pd.bdate_range("2024-01-01", periods=5)
    raw = {
        "AAA": _make_cached_frame(idx),
        "BBB": _make_cached_frame(idx),
    }
    wide = data_mod.to_wide_panel(raw)
    assert set(wide.keys()) >= {"open", "close", "adj_close", "volume"}
    assert wide["adj_close"].shape == (5, 2)
    assert list(wide["adj_close"].columns) == ["AAA", "BBB"]
