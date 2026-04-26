"""Cost model unit tests (deterministic; no API)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_agent.cost_models import (
    CompositeCostModel,
    FlatCostModel,
    SqrtImpactCostModel,
    cost_model_from_ips,
)
from quant_agent.ips import CostModelPolicy


@pytest.fixture
def small_panel():
    idx = pd.bdate_range("2022-01-01", periods=5)
    cols = ["A", "B", "C"]
    weights = pd.DataFrame([[0, 0, 0],
                            [0.5, -0.5, 0],
                            [0.5, -0.5, 0],
                            [0.0, -0.5, 0.5],
                            [0.0, -0.5, 0.5]], index=idx, columns=cols, dtype=float)
    prior = weights.shift(1).fillna(0.0)
    dollar_vol = pd.DataFrame(1e7, index=idx, columns=cols)
    return weights, prior, dollar_vol


def test_flat_cost_zero_when_no_change(small_panel):
    weights, _, _ = small_panel
    prior = weights.copy()  # no change
    cm = FlatCostModel(bps=5.0)
    cost = cm.cost_per_day(weights, prior)
    assert (cost == 0).all()


def test_flat_cost_proportional_to_turnover(small_panel):
    weights, prior, _ = small_panel
    cm = FlatCostModel(bps=10.0)
    cost = cm.cost_per_day(weights, prior)
    # Day 1: turnover = (0.5 + 0.5)/2 = 0.5 → 0.5 * 10/10000 = 0.0005
    assert cost.iloc[1] == pytest.approx(0.0005)
    # Days 2 + 4: no change → 0
    assert cost.iloc[2] == 0
    assert cost.iloc[4] == 0


def test_sqrt_impact_requires_dollar_vol(small_panel):
    weights, prior, _ = small_panel
    cm = SqrtImpactCostModel()
    with pytest.raises(ValueError, match="dollar_vol"):
        cm.cost_per_day(weights, prior, dollar_vol=None)


def test_sqrt_impact_returns_positive_for_trades(small_panel):
    weights, prior, dvol = small_panel
    cm = SqrtImpactCostModel(half_spread_bps=1.0, impact_coefficient=10.0)
    cost = cm.cost_per_day(weights, prior, dollar_vol=dvol, capital=1e8)
    # Days with trades (1, 3) should have positive cost
    assert cost.iloc[1] > 0
    assert cost.iloc[3] > 0


def test_sqrt_impact_scales_with_capital(small_panel):
    weights, prior, dvol = small_panel
    cm = SqrtImpactCostModel()
    small = cm.cost_per_day(weights, prior, dvol, capital=1e7)
    big = cm.cost_per_day(weights, prior, dvol, capital=1e9)
    # Bigger capital → bigger trade-to-ADV ratio → more impact cost
    assert big.iloc[1] > small.iloc[1]


def test_composite_sums_models(small_panel):
    weights, prior, dvol = small_panel
    flat = FlatCostModel(bps=5.0)
    impact = SqrtImpactCostModel()
    composite = CompositeCostModel(models=[flat, impact])
    flat_cost = flat.cost_per_day(weights, prior)
    impact_cost = impact.cost_per_day(weights, prior, dvol)
    comp_cost = composite.cost_per_day(weights, prior, dvol)
    pd.testing.assert_series_equal(comp_cost, flat_cost + impact_cost, check_names=False)


def test_factory_from_ips_policy():
    flat_policy = CostModelPolicy(type="flat", flat_bps=7.0)
    cm = cost_model_from_ips(flat_policy)
    assert isinstance(cm, FlatCostModel)
    assert cm.bps == 7.0

    sqrt_policy = CostModelPolicy(type="sqrt_impact", half_spread_bps=2.0, impact_coefficient=15.0)
    cm = cost_model_from_ips(sqrt_policy)
    assert isinstance(cm, SqrtImpactCostModel)
    assert cm.half_spread_bps == 2.0
    assert cm.impact_coefficient == 15.0


def test_factory_rejects_unknown_type():
    bad = CostModelPolicy(type="unknown")
    with pytest.raises(ValueError, match="unknown cost model type"):
        cost_model_from_ips(bad)
