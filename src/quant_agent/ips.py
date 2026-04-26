"""Investment Policy Statement — the constraints document that bounds the
multi-agent system, modelled on what an institutional fund's IPS does for
human PMs.

Conceptually the IPS is the contract the agents are operating under:
  * what universe and what data window they may use
  * what risk / exposure / turnover limits any candidate portfolio must respect
  * what cost model is binding for net-of-cost evaluation
  * what holdout window the meta-agent must use for promotion decisions
  * what governance rules the critic must enforce as hard vetoes

A YAML IPS is loaded once at the top of an orchestrator run and read by every
agent. Hard rules (`hard_constraints`) are non-negotiable veto triggers; soft
rules (`soft_constraints`) are guidance the agents should weigh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ----- schema ---------------------------------------------------------------


@dataclass
class UniversePolicy:
    source: str = "wikipedia"           # wikipedia | csv | api
    symbol_set: str = "sp500"
    point_in_time: bool = True
    min_dollar_vol_usd: float = 5_000_000


@dataclass
class DataPolicy:
    start: str = "2014-01-01"
    end: str = "today"
    fields: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")


@dataclass
class HoldoutPolicy:
    """The window the meta-agent uses to gate prompt promotions.

    Agents *may* see the in-sample window for proposal and exploration. The
    holdout window is reserved for promotion decisions; if a prompt rewrite
    does not improve OOS metrics on the holdout, it is not promoted.
    """
    in_sample_start: str = "2014-01-01"
    in_sample_end: str = "2021-12-31"
    holdout_start: str = "2022-01-01"
    holdout_end: str = "2026-04-17"
    min_observations: int = 252           # min trading days in holdout to act
    bonferroni_n_tests: int = 20          # divisor for multi-testing correction


@dataclass
class CostModelPolicy:
    type: str = "sqrt_impact"             # flat | sqrt_impact | composite
    half_spread_bps: float = 1.0
    impact_coefficient: float = 10.0      # bps per sqrt(trade/ADV)
    flat_bps: float = 5.0                 # used iff type == "flat"


@dataclass
class HardConstraint:
    """Non-negotiable veto. If a candidate portfolio violates, it is rejected."""
    name: str
    metric: str                            # e.g. "max_sector_weight" | "gross_exposure" | "avg_turnover"
    op: str                                # "<=" | "<" | ">=" | ">"
    threshold: float
    description: str = ""


@dataclass
class SoftConstraint:
    """Guidance — agents weigh but do not necessarily reject for violation."""
    name: str
    metric: str
    target: float
    weight: float = 1.0
    description: str = ""


@dataclass
class GovernancePolicy:
    """Rules the critic enforces."""
    require_oos_evaluation: bool = True
    multi_testing_correction: bool = True
    flag_inflated_sharpe_threshold: float = 0.5    # standalone Sharpe above this triggers human review
    flag_inflated_ic_ir_threshold: float = 2.5     # IC IR above this triggers look-ahead audit


@dataclass
class IPS:
    """The complete Investment Policy Statement."""
    name: str = "default"
    description: str = ""
    universe: UniversePolicy = field(default_factory=UniversePolicy)
    data: DataPolicy = field(default_factory=DataPolicy)
    holdout: HoldoutPolicy = field(default_factory=HoldoutPolicy)
    cost_model: CostModelPolicy = field(default_factory=CostModelPolicy)
    hard_constraints: list[HardConstraint] = field(default_factory=list)
    soft_constraints: list[SoftConstraint] = field(default_factory=list)
    governance: GovernancePolicy = field(default_factory=GovernancePolicy)


# ----- loader ---------------------------------------------------------------


def load_ips(path: str | Path) -> IPS:
    """Load an IPS from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(raw)


def _from_dict(d: dict[str, Any]) -> IPS:
    universe = UniversePolicy(**d.get("universe", {}))
    data = DataPolicy(**d.get("data", {}))
    holdout = HoldoutPolicy(**d.get("holdout", {}))
    cost_model = CostModelPolicy(**d.get("cost_model", {}))
    governance = GovernancePolicy(**d.get("governance", {}))
    hard = [HardConstraint(**hc) for hc in d.get("hard_constraints", [])]
    soft = [SoftConstraint(**sc) for sc in d.get("soft_constraints", [])]
    return IPS(
        name=d.get("name", "default"),
        description=d.get("description", ""),
        universe=universe,
        data=data,
        holdout=holdout,
        cost_model=cost_model,
        hard_constraints=hard,
        soft_constraints=soft,
        governance=governance,
    )


# ----- validation -----------------------------------------------------------


_KNOWN_HARD_METRICS = {
    "gross_exposure",
    "net_exposure",
    "avg_turnover",
    "max_drawdown",
    "max_sector_weight",
    "max_single_name_weight",
    "min_n_long",
    "min_n_short",
}

_KNOWN_OPS = {"<=", "<", ">=", ">"}


def validate_ips(ips: IPS) -> list[str]:
    """Return a list of validation errors. Empty list = valid."""
    errors: list[str] = []

    # Holdout window sanity
    if ips.holdout.in_sample_end >= ips.holdout.holdout_start:
        # String comparison works for ISO dates.
        errors.append(
            f"holdout.in_sample_end ({ips.holdout.in_sample_end}) must be < "
            f"holdout.holdout_start ({ips.holdout.holdout_start})"
        )

    # Cost model fields
    if ips.cost_model.type == "sqrt_impact":
        if ips.cost_model.half_spread_bps < 0:
            errors.append("cost_model.half_spread_bps must be >= 0")
        if ips.cost_model.impact_coefficient < 0:
            errors.append("cost_model.impact_coefficient must be >= 0")
    elif ips.cost_model.type == "flat":
        if ips.cost_model.flat_bps < 0:
            errors.append("cost_model.flat_bps must be >= 0")
    elif ips.cost_model.type == "composite":
        pass  # combined rules
    else:
        errors.append(
            f"cost_model.type must be one of flat | sqrt_impact | composite, got {ips.cost_model.type!r}"
        )

    # Hard constraints
    for hc in ips.hard_constraints:
        if hc.metric not in _KNOWN_HARD_METRICS:
            errors.append(f"hard_constraint {hc.name!r}: unknown metric {hc.metric!r}")
        if hc.op not in _KNOWN_OPS:
            errors.append(f"hard_constraint {hc.name!r}: op must be in {_KNOWN_OPS}, got {hc.op!r}")

    # Governance sanity
    if ips.governance.flag_inflated_sharpe_threshold <= 0:
        errors.append("governance.flag_inflated_sharpe_threshold must be > 0")
    if ips.governance.flag_inflated_ic_ir_threshold <= 0:
        errors.append("governance.flag_inflated_ic_ir_threshold must be > 0")

    return errors


def to_yaml_summary(ips: IPS) -> str:
    """Compact human-readable summary for inclusion in agent system prompts."""
    lines = [
        f"# IPS: {ips.name}",
        f"# {ips.description}" if ips.description else "",
        f"Universe: {ips.universe.symbol_set}, point_in_time={ips.universe.point_in_time}, "
        f"min_dollar_vol_usd={ips.universe.min_dollar_vol_usd:,}",
        f"Data window: {ips.data.start} to {ips.data.end}",
        f"In-sample: {ips.holdout.in_sample_start} to {ips.holdout.in_sample_end}",
        f"Holdout (frozen): {ips.holdout.holdout_start} to {ips.holdout.holdout_end} "
        f"(min {ips.holdout.min_observations} obs)",
        f"Cost model: {ips.cost_model.type} "
        f"(half_spread={ips.cost_model.half_spread_bps}bps, k={ips.cost_model.impact_coefficient}bps)"
        if ips.cost_model.type == "sqrt_impact"
        else f"Cost model: flat {ips.cost_model.flat_bps}bps",
        "",
        "Hard constraints (auto-veto on violation):",
    ]
    for hc in ips.hard_constraints:
        lines.append(f"  - {hc.name}: {hc.metric} {hc.op} {hc.threshold}")
    if not ips.hard_constraints:
        lines.append("  (none)")
    lines.append("")
    lines.append("Soft constraints (guidance):")
    for sc in ips.soft_constraints:
        lines.append(f"  - {sc.name}: {sc.metric} → {sc.target} (weight={sc.weight})")
    if not ips.soft_constraints:
        lines.append("  (none)")
    lines.append("")
    lines.append("Governance:")
    lines.append(f"  - require_oos_evaluation: {ips.governance.require_oos_evaluation}")
    lines.append(f"  - multi_testing_correction: {ips.governance.multi_testing_correction}")
    lines.append(
        f"  - audit if standalone Sharpe > {ips.governance.flag_inflated_sharpe_threshold}"
    )
    lines.append(
        f"  - audit if IC IR > {ips.governance.flag_inflated_ic_ir_threshold}"
    )
    return "\n".join([line for line in lines if line is not None])
