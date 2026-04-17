"""Journal persistence tests — no Anthropic API calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_agent.journal import Journal, build_state_recap


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(root=tmp_path / "research")


def test_save_and_load_feature_round_trip(journal):
    src = (
        "def my_feat(panel):\n"
        "    return panel['adj_close'].pct_change(5)\n"
    )
    meta = journal.save_feature("my_feat", src, "5d return")
    assert meta["name"] == "my_feat"
    assert meta["description"] == "5d return"
    assert meta["revision_count"] == 0

    loaded = journal.load_features()
    assert len(loaded) == 1
    assert loaded[0].name == "my_feat"
    assert loaded[0].source == src
    assert loaded[0].meta["description"] == "5d return"


def test_save_feature_bumps_revision_count_on_overwrite(journal):
    src1 = "def f(panel): return panel['adj_close']\n"
    src2 = "def f(panel): return -panel['adj_close']\n"
    journal.save_feature("f", src1, "v1")
    meta = journal.save_feature("f", src2, "v2 better")
    assert meta["revision_count"] == 1
    # File on disk has the new source.
    py_path = journal.root / "features" / "f.py"
    assert py_path.read_text() == src2


def test_reload_rejects_unsafe_source_but_keeps_session_alive(journal):
    journal._ensure()
    # Good feature.
    (journal.root / "features" / "good.py").write_text(
        "def good(panel): return panel['adj_close']\n"
    )
    (journal.root / "features" / "good.json").write_text(
        json.dumps({"name": "good", "description": "ok"})
    )
    # Someone hand-edits in a bad one.
    (journal.root / "features" / "evil.py").write_text(
        "import os\ndef evil(panel): os.system('rm -rf /'); return panel['adj_close']\n"
    )
    (journal.root / "features" / "evil.json").write_text(
        json.dumps({"name": "evil", "description": "nope"})
    )

    loaded = journal.load_features()
    names = [lf.name for lf in loaded]
    assert "good" in names
    assert "evil" not in names
    assert any("evil" in w for w in journal.load_warnings)


def test_record_run_and_best_updates(journal):
    r1 = journal.record_run(
        config={"feature_weights": {"mom_12_1": 1.0}, "halflife_days": 3, "weighting": "decile"},
        summary={"sharpe": 0.10, "ic_ir": 1.0, "gross_sharpe": 0.30},
    )
    r2 = journal.record_run(
        config={"feature_weights": {"mom_12_1": 1.0}, "halflife_days": 5, "weighting": "decile"},
        summary={"sharpe": 0.20, "ic_ir": 1.1, "gross_sharpe": 0.35},
    )
    assert r1 != r2

    best = journal.best()
    assert best is not None
    assert best["summary"]["sharpe"] == 0.20
    assert best["run_id"] == r2

    # A worse run does not dethrone the best.
    journal.record_run(
        config={"feature_weights": {"mom_12_1": 1.0}, "halflife_days": 7, "weighting": "decile"},
        summary={"sharpe": 0.05},
    )
    best2 = journal.best()
    assert best2["run_id"] == r2

    runs = journal.all_runs()
    assert len(runs) == 3
    # Preserves insertion order.
    assert runs[0]["run_id"] == r1
    assert runs[1]["run_id"] == r2


def test_record_run_updates_per_feature_best(journal):
    journal.save_feature("f1", "def f1(p): return p['adj_close']\n", "desc")
    journal.record_run(
        config={"feature_weights": {"f1": 1.0}},
        summary={"sharpe": 0.15},
    )
    meta = journal.feature_metadata("f1")
    assert meta["best_sharpe"] == 0.15

    # Lower Sharpe does not overwrite.
    journal.record_run(
        config={"feature_weights": {"f1": 1.0}},
        summary={"sharpe": 0.10},
    )
    meta = journal.feature_metadata("f1")
    assert meta["best_sharpe"] == 0.15

    # Higher Sharpe does.
    journal.record_run(
        config={"feature_weights": {"f1": 1.0}},
        summary={"sharpe": 0.25},
    )
    meta = journal.feature_metadata("f1")
    assert meta["best_sharpe"] == 0.25


def test_top_runs_ordering(journal):
    for i, s in enumerate([0.1, 0.3, 0.2, 0.05, 0.25]):
        journal.record_run(
            config={"feature_weights": {"mom_12_1": 1.0}, "iteration": i},
            summary={"sharpe": s},
        )
    top = journal.top_runs(3)
    assert [r["summary"]["sharpe"] for r in top] == [0.3, 0.25, 0.2]


def test_total_runs_counts_correctly(journal):
    assert journal.total_runs() == 0
    journal.record_run(config={"feature_weights": {}}, summary={"sharpe": 0.1})
    journal.record_run(config={"feature_weights": {}}, summary={"sharpe": 0.2})
    assert journal.total_runs() == 2


def test_clear_wipes_everything(journal):
    journal.save_feature("f1", "def f1(p): return p['adj_close']\n", "desc")
    journal.record_run(config={}, summary={"sharpe": 0.1})
    assert journal.root.exists()
    journal.clear()
    assert not journal.root.exists()


def test_build_state_recap_without_journal_data(journal):
    recap = build_state_recap(journal)
    assert "No previous runs" in recap


def test_build_state_recap_includes_best_and_features(journal):
    journal.save_feature("f1", "def f1(p): return p['adj_close']\n", "first feature")
    journal.record_run(
        config={
            "feature_weights": {"f1": 1.0, "mom_12_1": 1.0},
            "halflife_days": 3,
            "weighting": "decile_sticky",
        },
        summary={
            "sharpe": 0.14,
            "gross_sharpe": 0.30,
            "ic_ir": 1.2,
            "avg_turnover": 0.15,
            "max_drawdown": -0.45,
        },
    )
    recap = build_state_recap(journal)
    assert "Current best" in recap
    assert "f1" in recap
    assert "first feature" in recap
    assert "+0.140" in recap  # net sharpe formatted
