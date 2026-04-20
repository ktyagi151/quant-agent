"""Unit tests for the self-calibration loop."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_agent.calibration import CalibrationStore, build_calibration_recap


@pytest.fixture
def store(tmp_path: Path) -> CalibrationStore:
    return CalibrationStore(path=tmp_path / "predictions.jsonl")


def test_record_prediction_round_trip(store):
    uid = store.record("correlation", ["a", "b"], -0.2, 0.1, note="weak")
    assert uid
    rec = store.all_records()
    assert len(rec) == 1
    assert rec[0]["status"] == "pending"
    assert rec[0]["low"] == -0.2
    assert rec[0]["high"] == 0.1
    assert rec[0]["key"] == ["a", "b"]


def test_record_swaps_low_high_if_out_of_order(store):
    store.record("correlation", ["a", "b"], 0.5, -0.2)
    rec = store.all_records()[0]
    assert rec["low"] == -0.2
    assert rec["high"] == 0.5


def test_resolve_correlations_hits(store):
    store.record("correlation", ["a", "b"], -0.2, 0.1)
    store.record("correlation", ["c", "d"], 0.5, 0.9)  # pending, no data
    matrix = {"a": {"b": -0.05}, "b": {"a": -0.05}}
    resolved = store.resolve_correlations(matrix)
    assert len(resolved) == 1
    assert resolved[0]["in_range"] is True
    assert resolved[0]["actual"] == -0.05
    # Second prediction still pending.
    recs = store.all_records()
    pending = [r for r in recs if r["status"] == "pending"]
    assert len(pending) == 1


def test_resolve_correlations_misses(store):
    store.record("correlation", ["a", "b"], -0.2, 0.1)
    matrix = {"a": {"b": 0.5}}
    resolved = store.resolve_correlations(matrix)
    assert len(resolved) == 1
    assert resolved[0]["in_range"] is False
    assert resolved[0]["miss"] > 0  # actual above midpoint → underestimated


def test_resolve_correlations_handles_reversed_pair_order(store):
    store.record("correlation", ["a", "b"], -0.2, 0.1)
    # Matrix only has b -> a direction.
    matrix = {"b": {"a": -0.05}}
    resolved = store.resolve_correlations(matrix)
    assert len(resolved) == 1
    assert resolved[0]["actual"] == -0.05


def test_resolve_backtest(store):
    store.record("ic_ir", "baseline", 1.0, 1.5, note="prior")
    store.record("turnover", "baseline", 0.1, 0.2)
    store.record("net_sharpe", "baseline", 0.1, 0.2)
    summary = {"ic_ir": 1.3, "avg_turnover": 0.15, "sharpe": 0.18}
    resolved = store.resolve_backtest(summary, run_id="abc123")
    assert len(resolved) == 3
    assert all(r["in_range"] for r in resolved)
    for r in resolved:
        assert r["run_id"] == "abc123"


def test_resolve_feature_stats_by_name(store):
    store.record("rank_autocorr", "my_feat", 0.85, 0.95)
    store.record("rank_autocorr", "other_feat", 0.5, 0.7)
    stats = {"rank_autocorr_1d_mean": 0.90}
    resolved = store.resolve_feature_stats("my_feat", stats)
    assert len(resolved) == 1
    assert resolved[0]["in_range"] is True
    # The other prediction must still be pending.
    recs = store.all_records()
    pending = [r for r in recs if r["status"] == "pending"]
    assert len(pending) == 1
    assert pending[0]["key"] == "other_feat"


def test_summary_counts_and_bias(store):
    # Three corr predictions: 2 hits, 1 miss where actual is above.
    store.record("correlation", ["a", "b"], -0.1, 0.1)
    store.record("correlation", ["c", "d"], -0.1, 0.1)
    store.record("correlation", ["e", "f"], -0.1, 0.1)
    matrix = {
        "a": {"b": 0.05},
        "c": {"d": 0.00},
        "e": {"f": 0.30},  # miss, above range
    }
    store.resolve_correlations(matrix)
    s = store.summary()
    assert s["n_resolved"] == 3
    assert s["by_type"]["correlation"]["n"] == 3
    assert s["by_type"]["correlation"]["hit_rate"] == pytest.approx(2 / 3, abs=1e-2)
    # Mean signed miss across all three: (0.05 + 0.00 + 0.30)/3 ≈ +0.117
    assert s["by_type"]["correlation"]["mean_signed_miss_all"] == pytest.approx(0.117, abs=1e-2)


def test_recent_resolved_ordering(store):
    for pair in [["a", "b"], ["c", "d"], ["e", "f"]]:
        store.record("correlation", pair, -0.1, 0.1)
    store.resolve_correlations({"a": {"b": 0.0}})
    store.resolve_correlations({"c": {"d": 0.0}})
    store.resolve_correlations({"e": {"f": 0.0}})
    recent = store.recent_resolved(2)
    assert len(recent) == 2
    # Most recent first.
    assert recent[0]["key"] == ["e", "f"] or recent[0]["key"] == ["c", "d"]


def test_build_calibration_recap_empty_store(tmp_path):
    store = CalibrationStore(path=tmp_path / "empty.jsonl")
    recap = build_calibration_recap(store)
    assert recap == ""


def test_build_calibration_recap_after_resolutions(store):
    # 2 ic_ir predictions, 1 hit, 1 miss.
    store.record("ic_ir", "cfg1", 1.0, 1.5)
    store.record("ic_ir", "cfg2", 1.0, 1.5)
    store.resolve_backtest({"ic_ir": 1.3})  # hit
    store.resolve_backtest({"ic_ir": 2.0})  # miss, above range
    recap = build_calibration_recap(store)
    assert "2 resolved" in recap
    assert "ic_ir" in recap
    assert "hit rate" in recap


def test_session_auto_resolves_predictions(synthetic_panel, tmp_path, monkeypatch):
    """End-to-end: session records a prediction, runs a tool, prediction resolves."""
    import pandas as pd
    from quant_agent import features as feat_mod
    from quant_agent.agent_tools import ResearchSession, build_tools
    from quant_agent.calibration import CalibrationStore
    from quant_agent.journal import Journal

    # Use tmp paths for both stores.
    journal = Journal(root=tmp_path / "r")
    cal = CalibrationStore(path=tmp_path / "r" / "predictions.jsonl")

    pan = {}
    for k, v in synthetic_panel.items():
        tiled = pd.concat([v] * 2, ignore_index=True)
        tiled.index = pd.bdate_range("2019-01-01", periods=len(tiled))
        pan[k] = tiled
    baseline = feat_mod.compute_features(
        pan,
        ["mom_12_1", "reversal_5d", "volume_shock", "dollar_vol_20d", "vol_21d", "amihud_20d"],
    )

    session = ResearchSession(
        panel=pan,
        feature_fns=dict(feat_mod.FEATURES),
        feature_cache=baseline,
        sectors=None,
        membership_mask=None,
        liquidity_threshold=0.0,
        cost_bps=5.0,
        n_deciles=5,
        journal=journal,
        calibration=cal,
    )
    tools = build_tools(session)
    record = next(t for t in tools if t.name == "record_prediction")
    run_bt = next(t for t in tools if t.name == "run_backtest_tool")
    corrs = next(t for t in tools if t.name == "feature_correlations")

    # Prediction: ic_ir in [0, 3].
    record(predictions=[{"type": "ic_ir", "key": "baseline", "low": 0.0, "high": 3.0}])
    # Prediction: correlation mom_12_1 with reversal_5d in [-0.1, 0.1].
    record(
        predictions=[
            {"type": "correlation", "key": ["mom_12_1", "reversal_5d"], "low": -0.1, "high": 0.1}
        ]
    )
    assert cal.summary()["n_pending"] == 2

    # Trigger corr resolution.
    out = corrs(feature_names=["mom_12_1", "reversal_5d"])
    out_dict = json.loads(out)
    assert any(res for res in out_dict.get("calibration_resolved", []))
    assert cal.summary()["n_pending"] == 1

    # Trigger backtest resolution.
    run_bt(
        feature_weights={"mom_12_1": 1.0},
        halflife_days=0,
        weighting="decile",
        exit_n_deciles=None,
    )
    assert cal.summary()["n_pending"] == 0
    assert cal.summary()["n_resolved"] == 2
