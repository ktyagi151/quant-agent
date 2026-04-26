"""Risk metric + IPS constraint check tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_agent.ips import IPS, HardConstraint
from quant_agent.risk import (
    ConstraintViolation,
    avg_turnover,
    check_hard_constraints,
    compute_all_metrics,
    drawdown_path,
    gross_exposure,
    max_drawdown,
    max_sector_weight,
    n_long,
    n_short,
    risk_report,
    sector_exposures,
    soft_constraint_score,
)


@pytest.fixture
def simple_weights():
    idx = pd.bdate_range("2022-01-01", periods=10)
    cols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(10, 4)) * 0.1
    # Demean to make dollar-neutral
    w = pd.DataFrame(raw - raw.mean(axis=1, keepdims=True), index=idx, columns=cols)
    return w


def test_gross_exposure_matches_l1(simple_weights):
    g = gross_exposure(simple_weights)
    assert (g >= 0).all()
    pd.testing.assert_series_equal(g, simple_weights.abs().sum(axis=1), check_names=False)


def test_n_long_n_short_complementary(simple_weights):
    nl = n_long(simple_weights)
    ns = n_short(simple_weights)
    # Total positions per day ≤ ncol
    assert ((nl + ns) <= simple_weights.shape[1]).all()


def test_drawdown_monotone_nonpositive():
    rets = pd.Series([0.1, -0.2, 0.05, -0.1])
    dd = drawdown_path(rets)
    assert (dd <= 1e-12).all()
    assert max_drawdown(rets) <= 0


def test_avg_turnover_zero_for_constant_weights():
    idx = pd.bdate_range("2022-01-01", periods=5)
    w = pd.DataFrame([[0.5, -0.5]] * 5, index=idx, columns=["A", "B"])
    assert avg_turnover(w) == pytest.approx(0)


def test_sector_exposures_groups_correctly():
    idx = pd.bdate_range("2022-01-01", periods=3)
    w = pd.DataFrame([[0.3, -0.2, 0.5, -0.4]] * 3, index=idx, columns=["A", "B", "C", "D"])
    sectors = pd.Series({"A": "X", "B": "X", "C": "Y", "D": "Y"})
    s = sector_exposures(w, sectors)
    assert s.shape == (3, 2)
    # X = |0.3| + |-0.2| = 0.5; Y = |0.5| + |-0.4| = 0.9
    assert s["X"].iloc[0] == pytest.approx(0.5)
    assert s["Y"].iloc[0] == pytest.approx(0.9)


def test_check_hard_constraints_passes():
    metrics = {"gross_exposure": 1.5, "max_single_name_weight": 0.03}
    ips = IPS(hard_constraints=[
        HardConstraint(name="g", metric="gross_exposure", op="<=", threshold=2.0),
        HardConstraint(name="n", metric="max_single_name_weight", op="<=", threshold=0.05),
    ])
    vios = check_hard_constraints(metrics, ips)
    assert vios == []


def test_check_hard_constraints_catches_violation():
    metrics = {"gross_exposure": 3.0}
    ips = IPS(hard_constraints=[
        HardConstraint(name="g", metric="gross_exposure", op="<=", threshold=2.0),
    ])
    vios = check_hard_constraints(metrics, ips)
    assert len(vios) == 1
    assert vios[0].constraint.name == "g"
    assert vios[0].actual == 3.0


def test_check_hard_constraints_drawdown_floor():
    """Max drawdown is non-positive; constraint is `>=` -0.50."""
    metrics = {"max_drawdown": -0.55}
    ips = IPS(hard_constraints=[
        HardConstraint(name="dd", metric="max_drawdown", op=">=", threshold=-0.50),
    ])
    vios = check_hard_constraints(metrics, ips)
    assert len(vios) == 1


def test_risk_report_passes_or_fails(simple_weights):
    sectors = pd.Series({c: "X" for c in simple_weights.columns})
    rets = simple_weights.iloc[:, 0]   # any series
    ips = IPS(hard_constraints=[
        HardConstraint(name="g", metric="gross_exposure", op="<=", threshold=10.0),
    ])
    rep = risk_report(simple_weights, rets, sectors, ips)
    assert rep["passes_ips"] is True
    assert "metrics" in rep
    assert "hard_violations" in rep
