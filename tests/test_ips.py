"""IPS schema, loader, validator, summary."""
from __future__ import annotations

from pathlib import Path

import pytest

from quant_agent.ips import (
    IPS,
    HardConstraint,
    HoldoutPolicy,
    SoftConstraint,
    load_ips,
    to_yaml_summary,
    validate_ips,
)


def test_default_ips_validates_clean():
    ips = IPS()
    errs = validate_ips(ips)
    assert errs == []


def test_loads_default_yaml():
    path = Path(__file__).parent.parent / "configs" / "ips" / "default.yaml"
    ips = load_ips(path)
    assert ips.name == "default"
    errs = validate_ips(ips)
    assert errs == []
    assert ips.cost_model.type == "sqrt_impact"
    assert any(hc.metric == "gross_exposure" for hc in ips.hard_constraints)


def test_validate_catches_holdout_overlap():
    ips = IPS(holdout=HoldoutPolicy(in_sample_end="2025-12-31", holdout_start="2024-01-01"))
    errs = validate_ips(ips)
    assert any("holdout" in e for e in errs)


def test_validate_catches_unknown_metric():
    ips = IPS(hard_constraints=[HardConstraint(name="bad", metric="unknown_metric", op="<=", threshold=1.0)])
    errs = validate_ips(ips)
    assert any("unknown metric" in e for e in errs)


def test_validate_catches_bad_op():
    ips = IPS(hard_constraints=[HardConstraint(name="bad", metric="gross_exposure", op="==", threshold=1.0)])
    errs = validate_ips(ips)
    assert any("op must be" in e for e in errs)


def test_summary_renders_all_sections():
    ips = IPS(
        hard_constraints=[HardConstraint(name="g", metric="gross_exposure", op="<=", threshold=2.0)],
        soft_constraints=[SoftConstraint(name="t", metric="avg_turnover", target=0.2, weight=1.0)],
    )
    s = to_yaml_summary(ips)
    assert "Hard constraints" in s
    assert "Soft constraints" in s
    assert "Governance" in s
    assert "gross_exposure" in s
