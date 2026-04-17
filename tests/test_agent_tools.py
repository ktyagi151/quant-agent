"""Session-level tests — no Anthropic API calls."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_agent import features as feat_mod
from quant_agent.agent_tools import ResearchSession, build_tools


@pytest.fixture
def tiny_session(synthetic_panel) -> ResearchSession:
    # Extend the panel to 400 rows so baseline features (mom_12_1 needs 252 lookback) are non-trivial.
    pan = {}
    for k, v in synthetic_panel.items():
        # Tile rows to 400 for enough history.
        tiled = pd.concat([v] * 2, ignore_index=True)
        tiled.index = pd.bdate_range("2019-01-01", periods=len(tiled))
        pan[k] = tiled

    baseline = feat_mod.compute_features(
        pan,
        ["mom_12_1", "reversal_5d", "volume_shock", "dollar_vol_20d", "vol_21d", "amihud_20d"],
    )
    sectors = pd.Series(
        {t: ("A" if i % 2 == 0 else "B") for i, t in enumerate(pan["adj_close"].columns)}
    )
    # No membership mask for the tiny test (all included).
    return ResearchSession(
        panel=pan,
        feature_fns=dict(feat_mod.FEATURES),
        feature_cache=baseline,
        sectors=sectors,
        membership_mask=None,
        liquidity_threshold=0.0,
        cost_bps=5.0,
        n_deciles=5,  # fewer buckets because we only have 8 tickers
    )


def test_run_backtest_with_baseline_features(tiny_session):
    summary = tiny_session.run(
        feature_weights={"mom_12_1": 1.0, "reversal_5d": 1.0},
        halflife_days=3,
        weighting="decile_sticky",
        exit_n_deciles=3,
    )
    assert "sharpe" in summary
    assert "gross_sharpe" in summary
    assert "ic_ir" in summary
    assert summary["n_obs"] > 100
    assert len(tiny_session.history) == 1


def test_propose_feature_via_tool_runs_it(tiny_session):
    _list, propose, run_bt = build_tools(tiny_session)
    code = (
        "def range_feature(panel):\n"
        "    high_low = (panel['high'] - panel['low']) / panel['close']\n"
        "    return -high_low.rolling(10).mean()\n"
    )
    msg = propose(name="range_feature", python_code=code, description="negative 10d HL range")
    assert msg.startswith("OK")
    assert "range_feature" in tiny_session.feature_fns

    result_json = run_bt(
        feature_weights={"mom_12_1": 1.0, "range_feature": 0.5},
        halflife_days=2,
        weighting="decile_sticky",
        exit_n_deciles=3,
    )
    result = json.loads(result_json)
    assert "sharpe" in result


def test_propose_feature_unsafe_rejected(tiny_session):
    _list, propose, _run_bt = build_tools(tiny_session)
    msg = propose(
        name="bad",
        python_code="import os\ndef bad(p): return p['adj_close']",
        description="evil",
    )
    assert msg.startswith("REJECTED")
    assert "bad" not in tiny_session.feature_fns


def test_propose_feature_wrong_shape_rejected(tiny_session):
    _list, propose, _run_bt = build_tools(tiny_session)
    # Returns a Series, not a DataFrame.
    msg = propose(
        name="wrong_shape",
        python_code="def wrong_shape(p): return p['adj_close'].mean(axis=1)",
        description="collapses to series",
    )
    assert msg.startswith("REJECTED")
    assert "wrong_shape" not in tiny_session.feature_fns


def test_list_features_returns_sorted_json(tiny_session):
    list_feats, _propose, _run_bt = build_tools(tiny_session)
    out = json.loads(list_feats())
    assert isinstance(out, list)
    assert "mom_12_1" in out
    assert out == sorted(out)


def test_run_backtest_unknown_feature_errors(tiny_session):
    _list, _propose, run_bt = build_tools(tiny_session)
    msg = run_bt(
        feature_weights={"nonexistent_feature": 1.0},
        halflife_days=3,
        weighting="decile",
        exit_n_deciles=5,
    )
    assert "ERROR" in msg
    assert "nonexistent_feature" in msg


# ----- journal integration --------------------------------------------------


def _inject_journal(session, journal) -> None:
    """Attach a journal to an existing session for integration tests."""
    session.journal = journal


def test_session_persists_features_to_journal(tiny_session, tmp_path):
    from quant_agent.journal import Journal

    journal = Journal(root=tmp_path / "r")
    _inject_journal(tiny_session, journal)
    _list, propose, _run_bt = build_tools(tiny_session)

    code = "def hl_range(panel): return -(panel['high'] - panel['low']) / panel['close']\n"
    msg = propose(name="hl_range", python_code=code, description="negative HL range")
    assert msg.startswith("OK")

    # Feature is persisted on disk.
    assert (journal.root / "features" / "hl_range.py").exists()
    assert (journal.root / "features" / "hl_range.json").exists()

    meta = journal.feature_metadata("hl_range")
    assert meta["description"] == "negative HL range"


def test_session_records_runs_and_updates_best(tiny_session, tmp_path):
    from quant_agent.journal import Journal

    journal = Journal(root=tmp_path / "r")
    _inject_journal(tiny_session, journal)

    summary = tiny_session.run(
        feature_weights={"mom_12_1": 1.0, "reversal_5d": 1.0},
        halflife_days=3,
        weighting="decile_sticky",
        exit_n_deciles=3,
    )
    assert "run_id" in summary
    assert journal.total_runs() == 1

    best = journal.best()
    assert best is not None
    assert best["run_id"] == summary["run_id"]


def test_session_reloads_persisted_features_from_journal(tmp_path, synthetic_panel):
    """A fresh session with an existing journal picks up prior features."""
    import pandas as pd
    from quant_agent import features as feat_mod
    from quant_agent.agent_tools import ResearchSession
    from quant_agent.journal import Journal

    journal = Journal(root=tmp_path / "r")
    src = "def my_custom(panel): return panel['adj_close'].pct_change(3)\n"
    journal.save_feature("my_custom", src, "3d pct change")

    # Build a session pointing at this journal (bypass from_cache which needs full data).
    pan = {}
    for k, v in synthetic_panel.items():
        tiled = pd.concat([v] * 2, ignore_index=True)
        tiled.index = pd.bdate_range("2019-01-01", periods=len(tiled))
        pan[k] = tiled
    baseline = feat_mod.compute_features(
        pan,
        ["mom_12_1", "reversal_5d", "volume_shock", "dollar_vol_20d", "vol_21d", "amihud_20d"],
    )

    # Simulate the from_cache branch that loads journal features.
    feature_fns = dict(feat_mod.FEATURES)
    feature_cache = dict(baseline)
    loaded = journal.load_features()
    for lf in loaded:
        feature_fns[lf.name] = lf.fn
        feature_cache[lf.name] = lf.fn(pan)

    session = ResearchSession(
        panel=pan,
        feature_fns=feature_fns,
        feature_cache=feature_cache,
        sectors=None,
        membership_mask=None,
        liquidity_threshold=0.0,
        cost_bps=5.0,
        n_deciles=5,
        journal=journal,
    )

    assert "my_custom" in session.feature_fns
    # And can actually run with it.
    summary = session.run(
        feature_weights={"mom_12_1": 1.0, "my_custom": 0.5},
        halflife_days=2,
        weighting="decile",
        exit_n_deciles=None,
    )
    assert "sharpe" in summary
    # Run was recorded.
    assert journal.total_runs() == 1
