"""Pluggable transaction-cost models.

The current backtest applies a flat-bps × turnover charge, which is the
simplest and most pessimistic model. For larger names with deep liquidity,
a square-root impact model is standard:

  cost(t, ticker) = (half_spread + impact_coef · sqrt(|trade$| / ADV$)) bps

where:
  - half_spread is the no-trade cost (e.g. 1bps for SP500 large caps)
  - impact_coef is the impact per √(participation rate) (typical: 5–20bps)
  - trade$ is the dollar size of the trade
  - ADV$ is the rolling average dollar volume

This module exposes a small interface that the backtest can call: given a
weights DataFrame, prior weights, and a panel, return a per-date scalar
cost in fractional return units (i.e. cost=0.0005 = 5bps).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ips import CostModelPolicy


# ----- interface ------------------------------------------------------------


class CostModel(ABC):
    """Abstract base — concrete subclasses implement `cost_per_day`."""

    @abstractmethod
    def cost_per_day(
        self,
        weights: pd.DataFrame,
        prior_weights: pd.DataFrame,
        dollar_vol: pd.DataFrame | None = None,
        capital: float = 1.0,
    ) -> pd.Series:
        """Return per-date cost (in fractional return units).

        Args:
            weights: DataFrame[date × ticker] of target weights for that day.
            prior_weights: same shape; weights at t-1.
            dollar_vol: DataFrame[date × ticker] of average dollar volume,
                only required for impact models.
            capital: AUM in dollars (sets the scale of the trade).
        """


# ----- flat ------------------------------------------------------------------


@dataclass
class FlatCostModel(CostModel):
    """Simple `cost = bps × turnover` model."""
    bps: float = 5.0

    def cost_per_day(
        self,
        weights: pd.DataFrame,
        prior_weights: pd.DataFrame,
        dollar_vol: pd.DataFrame | None = None,
        capital: float = 1.0,
    ) -> pd.Series:
        dw = (weights - prior_weights).abs()
        # Half-turnover: standard "buy + sell counted once".
        turnover = dw.sum(axis=1) / 2.0
        return turnover * (self.bps / 1e4)


# ----- sqrt impact ----------------------------------------------------------


@dataclass
class SqrtImpactCostModel(CostModel):
    """Spread + sqrt(participation) impact, per ticker per day.

    This is the standard model from Almgren et al. (2005), "Direct Estimation
    of Equity Market Impact." Total cost in bps for trading a single name:

      bps = half_spread + impact_coef · sqrt(|trade$| / ADV$)

    For a portfolio:
      portfolio_cost_per_$ = sum_i |Δw_i| · bps_i

    where bps_i is the per-ticker cost (depends on the trade's ADV ratio).

    The model is parameterized to give realistic numbers:
      half_spread_bps ~ 1.0 for SP500 large caps
      impact_coef ~ 10 bps per √(participation rate)
    """
    half_spread_bps: float = 1.0
    impact_coefficient: float = 10.0
    min_dollar_vol: float = 1e6  # floor to prevent /0 on sparse ADV

    def cost_per_day(
        self,
        weights: pd.DataFrame,
        prior_weights: pd.DataFrame,
        dollar_vol: pd.DataFrame | None = None,
        capital: float = 1.0,
    ) -> pd.Series:
        if dollar_vol is None:
            raise ValueError("SqrtImpactCostModel requires `dollar_vol` (e.g. dollar_vol_20d)")

        dw = (weights - prior_weights).abs()
        trade_dollars = dw * capital                 # |Δw| · AUM
        adv = dollar_vol.reindex_like(dw).clip(lower=self.min_dollar_vol)

        # Per-ticker bps cost
        participation = trade_dollars / adv
        impact_bps = self.impact_coefficient * np.sqrt(participation)
        spread_bps = self.half_spread_bps  # constant per trade
        per_ticker_bps = (spread_bps + impact_bps).where(dw > 0, 0.0)

        # Per-date total cost = sum over tickers of |Δw_i| · bps_i / 10000
        cost = (dw * per_ticker_bps).sum(axis=1) / 1e4
        return cost


# ----- composite ------------------------------------------------------------


@dataclass
class CompositeCostModel(CostModel):
    """Sum of multiple cost models, e.g. flat + impact."""
    models: list[CostModel]

    def cost_per_day(
        self,
        weights: pd.DataFrame,
        prior_weights: pd.DataFrame,
        dollar_vol: pd.DataFrame | None = None,
        capital: float = 1.0,
    ) -> pd.Series:
        if not self.models:
            return pd.Series(0.0, index=weights.index)
        out = self.models[0].cost_per_day(weights, prior_weights, dollar_vol, capital)
        for m in self.models[1:]:
            out = out.add(m.cost_per_day(weights, prior_weights, dollar_vol, capital), fill_value=0.0)
        return out


# ----- factory --------------------------------------------------------------


def cost_model_from_ips(policy: CostModelPolicy) -> CostModel:
    """Construct the appropriate CostModel from an IPS policy block."""
    if policy.type == "flat":
        return FlatCostModel(bps=policy.flat_bps)
    if policy.type == "sqrt_impact":
        return SqrtImpactCostModel(
            half_spread_bps=policy.half_spread_bps,
            impact_coefficient=policy.impact_coefficient,
        )
    if policy.type == "composite":
        return CompositeCostModel(
            models=[
                FlatCostModel(bps=policy.flat_bps),
                SqrtImpactCostModel(
                    half_spread_bps=policy.half_spread_bps,
                    impact_coefficient=policy.impact_coefficient,
                ),
            ]
        )
    raise ValueError(f"unknown cost model type: {policy.type!r}")
