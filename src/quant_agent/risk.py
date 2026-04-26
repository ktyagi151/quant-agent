"""Risk monitoring + IPS hard-constraint enforcement.

This is the deterministic side of the cost/risk agent — given a weights
DataFrame and an IPS, compute every metric the IPS knows about and check
which hard constraints are violated. The agent layer reads these results
and either passes them to the critic for veto, or repairs the weights.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .ips import IPS, HardConstraint


# ----- exposure metrics -----------------------------------------------------


def gross_exposure(weights: pd.DataFrame) -> pd.Series:
    """Per-date sum of |w_i|."""
    return weights.abs().sum(axis=1)


def net_exposure(weights: pd.DataFrame) -> pd.Series:
    """Per-date |sum w_i|."""
    return weights.sum(axis=1).abs()


def sector_exposures(weights: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Per-date × sector aggregate |weight|. Returns date × sector DataFrame."""
    sect_map = sectors.reindex(weights.columns).fillna("UNKNOWN")
    abs_w = weights.abs()
    grouped = abs_w.T.groupby(sect_map).sum().T
    return grouped


def max_sector_weight(weights: pd.DataFrame, sectors: pd.Series) -> pd.Series:
    """Per-date max sector exposure."""
    s = sector_exposures(weights, sectors)
    return s.max(axis=1)


def max_single_name(weights: pd.DataFrame) -> pd.Series:
    """Per-date max |w_i| across all tickers."""
    return weights.abs().max(axis=1)


def n_long(weights: pd.DataFrame) -> pd.Series:
    """Per-date count of names with positive weight."""
    return (weights > 0).sum(axis=1)


def n_short(weights: pd.DataFrame) -> pd.Series:
    """Per-date count of names with negative weight."""
    return (weights < 0).sum(axis=1)


def avg_turnover(weights: pd.DataFrame) -> float:
    """Mean daily half-L1 turnover across the whole period."""
    dw = weights.diff().abs()
    return float((dw.sum(axis=1) / 2.0).mean())


def drawdown_path(returns: pd.Series) -> pd.Series:
    """Equity-curve drawdown at each date (≤ 0)."""
    eq = (1 + returns.fillna(0)).cumprod()
    peak = eq.cummax()
    return eq / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_path(returns).min())


# ----- aggregate metrics dict for IPS evaluation ---------------------------


def compute_all_metrics(
    weights: pd.DataFrame,
    returns: pd.Series | None = None,
    sectors: pd.Series | None = None,
) -> dict[str, float]:
    """Compute the worst-case (most-violated) value of every IPS-known metric.

    For per-date metrics, we use:
      - max over time for caps (e.g. gross_exposure, sector_weight)
      - min over time for breadth (e.g. min_n_long)
      - mean over time for turnover
      - drawdown peak-to-trough for max_drawdown

    These are the values the IPS hard-constraint check evaluates against.
    """
    metrics: dict[str, float] = {}
    metrics["gross_exposure"] = float(gross_exposure(weights).max())
    metrics["net_exposure"] = float(net_exposure(weights).max())
    metrics["max_single_name_weight"] = float(max_single_name(weights).max())
    metrics["min_n_long"] = float(n_long(weights).min())
    metrics["min_n_short"] = float(n_short(weights).min())
    metrics["avg_turnover"] = avg_turnover(weights)

    if sectors is not None:
        metrics["max_sector_weight"] = float(max_sector_weight(weights, sectors).max())

    if returns is not None:
        metrics["max_drawdown"] = max_drawdown(returns)

    return metrics


# ----- constraint check ----------------------------------------------------


_OPS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
}


@dataclass
class ConstraintViolation:
    constraint: HardConstraint
    actual: float

    def __str__(self) -> str:
        return (
            f"VIOLATION {self.constraint.name}: "
            f"{self.constraint.metric} = {self.actual:.4f}, "
            f"required {self.constraint.op} {self.constraint.threshold}"
        )


def check_hard_constraints(
    metrics: dict[str, float], ips: IPS
) -> list[ConstraintViolation]:
    """Return all hard constraints violated by `metrics`. Empty list = pass."""
    violations: list[ConstraintViolation] = []
    for hc in ips.hard_constraints:
        if hc.metric not in metrics:
            continue   # metric not measured — skip rather than silently fail
        actual = metrics[hc.metric]
        op = _OPS[hc.op]
        # The constraint is "metric op threshold". To check, we ask: does actual op threshold hold?
        if not op(actual, hc.threshold):
            violations.append(ConstraintViolation(constraint=hc, actual=actual))
    return violations


def soft_constraint_score(metrics: dict[str, float], ips: IPS) -> dict[str, float]:
    """Score how close each soft-constraint metric is to its target (0 = exact, lower is better).

    Returns per-constraint absolute deviation × weight. Useful for the critic
    to rank candidates that all pass hard constraints.
    """
    out: dict[str, float] = {}
    for sc in ips.soft_constraints:
        if sc.metric not in metrics:
            continue
        deviation = abs(metrics[sc.metric] - sc.target)
        out[sc.name] = deviation * sc.weight
    return out


# ----- summary report -------------------------------------------------------


def risk_report(
    weights: pd.DataFrame,
    returns: pd.Series | None,
    sectors: pd.Series | None,
    ips: IPS,
) -> dict:
    """Single dict combining all metrics + hard-constraint violations + soft-constraint scores."""
    m = compute_all_metrics(weights, returns=returns, sectors=sectors)
    vios = check_hard_constraints(m, ips)
    soft = soft_constraint_score(m, ips)
    return {
        "metrics": m,
        "hard_violations": [
            {"name": v.constraint.name, "metric": v.constraint.metric,
             "op": v.constraint.op, "threshold": v.constraint.threshold,
             "actual": v.actual}
            for v in vios
        ],
        "soft_scores": soft,
        "passes_ips": len(vios) == 0,
    }
