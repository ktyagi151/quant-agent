"""Versioned prompt store tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_agent.prompt_history import PromptHistory, PromptVersion


@pytest.fixture
def store(tmp_path: Path) -> PromptHistory:
    return PromptHistory(root=tmp_path / "prompts")


def test_add_proposal_returns_version(store):
    pv = store.add_proposal("alpha", "v1 text", "first version")
    assert pv.version == 1
    assert pv.parent_version is None
    assert pv.text_hash


def test_versions_increment(store):
    pv1 = store.add_proposal("alpha", "v1", "first")
    pv2 = store.add_proposal("alpha", "v2", "second")
    pv3 = store.add_proposal("alpha", "v3", "third")
    assert [v.version for v in store.list_versions("alpha")] == [1, 2, 3]
    assert pv2.parent_version == 1
    assert pv3.parent_version == 2


def test_promote_records_metrics(store):
    pv = store.add_proposal("alpha", "text", "rationale")
    store.promote(pv, metrics={"sharpe": 0.2, "ic_ir": 1.3})
    latest = store.latest("alpha")
    assert latest.promoted is True
    assert latest.metrics_at_promotion == {"sharpe": 0.2, "ic_ir": 1.3}


def test_rollback_marks_version(store):
    pv = store.add_proposal("alpha", "text", "rationale")
    store.promote(pv, metrics={"sharpe": 0.2})
    store.rollback(pv)
    latest = store.latest("alpha")
    assert latest.rolled_back is True
    assert latest.rolled_back_at is not None


def test_latest_promoted_skips_rolled_back(store):
    v1 = store.add_proposal("alpha", "v1 text", "first")
    store.promote(v1, metrics={"sharpe": 0.1})
    v2 = store.add_proposal("alpha", "v2 text", "second")
    store.promote(v2, metrics={"sharpe": 0.2})
    store.rollback(v2)
    latest_p = store.latest_promoted("alpha")
    assert latest_p is not None
    assert latest_p.version == 1


def test_diff_two_versions(store):
    v1 = store.add_proposal("alpha", "line1\nline2\nline3", "first")
    v2 = store.add_proposal("alpha", "line1\nlineX\nline3", "second")
    diff = store.diff("alpha", 1, 2)
    assert "line2" in diff
    assert "lineX" in diff


def test_isolation_per_agent(store):
    store.add_proposal("alpha", "alpha v1", "first")
    store.add_proposal("portfolio", "portfolio v1", "first")
    assert len(store.list_versions("alpha")) == 1
    assert len(store.list_versions("portfolio")) == 1
    # Independent counters
    pv2 = store.add_proposal("portfolio", "portfolio v2", "second")
    assert pv2.version == 2
    assert len(store.list_versions("alpha")) == 1
