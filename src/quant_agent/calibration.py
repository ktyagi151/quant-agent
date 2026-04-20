"""Self-calibration loop — track agent forecast accuracy across sessions.

The agent calls `record_prediction` before a tool call when it has a prior
(e.g. "correlation with mom_12_1 in [-0.2, -0.1]"). The prediction is stored
as pending. After the corresponding tool call runs, the relevant handler in
`ResearchSession` auto-resolves matching predictions by scanning the
tool's output. Over time, the agent's calibration track record appears in
the state recap so successive runs can see their own bias.

Supported prediction types (v1):
  * `correlation`       — key = [feature_a, feature_b], auto-resolved from feature_correlations
  * `rank_autocorr`     — key = feature_name, auto-resolved from feature_stats
  * `ic_ir`             — key = config label (free-form), auto-resolved from next backtest
  * `net_sharpe`        — key = config label, auto-resolved from next backtest
  * `gross_sharpe`      — key = config label, auto-resolved from next backtest
  * `turnover`          — key = config label, auto-resolved from next backtest

Resolution policy: pending records match against the NEXT observation of the
same type after they were recorded. Multiple pending records of the same
type and key are all resolved by the first matching observation.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .io_utils import data_dir


_BACKTEST_TYPES = {"ic_ir", "net_sharpe", "gross_sharpe", "turnover"}
_SUMMARY_FIELD = {
    "ic_ir": "ic_ir",
    "net_sharpe": "sharpe",
    "gross_sharpe": "gross_sharpe",
    "turnover": "avg_turnover",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_key(key) -> list[str] | str:
    """Correlations carry a 2-element list; scalar types carry a single string."""
    if isinstance(key, list):
        return [str(k) for k in key]
    return str(key)


@dataclass
class CalibrationStore:
    path: Path

    @classmethod
    def default(cls) -> "CalibrationStore":
        return cls(path=data_dir() / "research" / "predictions.jsonl")

    # ----- low-level IO ---------------------------------------------------

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _write_all(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ----- recording ------------------------------------------------------

    def record(
        self,
        type: str,
        key,
        low: float,
        high: float,
        note: str = "",
    ) -> str:
        if low > high:
            low, high = high, low  # accept either ordering
        uid = uuid.uuid4().hex[:8]
        record = {
            "id": uid,
            "timestamp": _now_iso(),
            "type": type,
            "key": _normalize_key(key),
            "low": float(low),
            "high": float(high),
            "note": note,
            "status": "pending",
        }
        self._append(record)
        return uid

    # ----- resolution -----------------------------------------------------

    def resolve_correlations(self, matrix: dict) -> list[dict]:
        """Resolve pending `correlation` predictions against a correlation matrix.

        matrix format: {feature_a: {feature_b: value, ...}, ...}
        """
        records = self._read()
        resolved: list[dict] = []
        for r in records:
            if r["status"] != "pending" or r["type"] != "correlation":
                continue
            key = r["key"]
            if not isinstance(key, list) or len(key) != 2:
                continue
            a, b = key
            val = None
            for x, y in ((a, b), (b, a)):
                row = matrix.get(x)
                if isinstance(row, dict):
                    v = row.get(y)
                    if v is not None:
                        val = v
                        break
            if val is None:
                continue
            self._mark_resolved(r, float(val))
            resolved.append(r)
        if resolved:
            self._write_all(records)
        return resolved

    def resolve_backtest(self, summary: dict, run_id: str | None = None) -> list[dict]:
        records = self._read()
        resolved: list[dict] = []
        for r in records:
            if r["status"] != "pending":
                continue
            if r["type"] not in _BACKTEST_TYPES:
                continue
            field_name = _SUMMARY_FIELD[r["type"]]
            val = summary.get(field_name)
            if val is None:
                continue
            self._mark_resolved(r, float(val), run_id=run_id)
            resolved.append(r)
        if resolved:
            self._write_all(records)
        return resolved

    def resolve_feature_stats(self, name: str, stats: dict) -> list[dict]:
        records = self._read()
        resolved: list[dict] = []
        for r in records:
            if r["status"] != "pending" or r["type"] != "rank_autocorr":
                continue
            key_match = r["key"] == name or r["key"] == [name]
            if not key_match:
                continue
            val = stats.get("rank_autocorr_1d_mean")
            if val is None:
                continue
            self._mark_resolved(r, float(val))
            resolved.append(r)
        if resolved:
            self._write_all(records)
        return resolved

    def _mark_resolved(self, r: dict, actual: float, run_id: str | None = None) -> None:
        r["status"] = "resolved"
        r["actual"] = actual
        r["in_range"] = r["low"] <= actual <= r["high"]
        mid = (r["low"] + r["high"]) / 2.0
        r["miss"] = actual - mid  # positive → actual above midpoint → agent underestimated
        r["resolved_at"] = _now_iso()
        if run_id is not None:
            r["run_id"] = run_id

    # ----- reporting ------------------------------------------------------

    def all_records(self) -> list[dict]:
        return self._read()

    def summary(self) -> dict:
        records = self._read()
        resolved = [r for r in records if r["status"] == "resolved"]
        pending = [r for r in records if r["status"] == "pending"]

        by_type: dict[str, dict] = {}
        for r in resolved:
            t = r["type"]
            by_type.setdefault(t, {"hits": 0, "total": 0, "misses": []})
            by_type[t]["total"] += 1
            if r.get("in_range"):
                by_type[t]["hits"] += 1
            else:
                by_type[t]["misses"].append(r.get("miss", 0.0))

        by_type_out: dict[str, dict] = {}
        for t, s in by_type.items():
            hit_rate = s["hits"] / s["total"] if s["total"] else 0.0
            mean_signed_miss_on_misses = (
                sum(s["misses"]) / len(s["misses"]) if s["misses"] else 0.0
            )
            # Bias across all resolved (including hits, where miss is inside range).
            all_miss = [
                (r["actual"] - (r["low"] + r["high"]) / 2)
                for r in resolved
                if r["type"] == t
            ]
            mean_signed_miss = sum(all_miss) / len(all_miss) if all_miss else 0.0
            by_type_out[t] = {
                "n": s["total"],
                "hit_rate": round(hit_rate, 3),
                "mean_signed_miss_all": round(mean_signed_miss, 3),
                "mean_signed_miss_on_misses": round(mean_signed_miss_on_misses, 3),
            }

        return {
            "n_total": len(records),
            "n_resolved": len(resolved),
            "n_pending": len(pending),
            "by_type": by_type_out,
        }

    def recent_resolved(self, k: int = 10) -> list[dict]:
        # File order is insertion order; the last records are the most recent.
        # resolved_at has second-granularity so we rely on file order for ties.
        resolved = [r for r in self._read() if r["status"] == "resolved"]
        return list(reversed(resolved))[:k]


# ----- state recap integration -----------------------------------------------


def build_calibration_recap(store: CalibrationStore, max_recent: int = 6) -> str:
    """Short block suitable for inclusion in the session state recap."""
    s = store.summary()
    if s["n_resolved"] == 0:
        return ""
    lines = [f"## Calibration track record ({s['n_resolved']} resolved predictions)"]
    for t in sorted(s["by_type"].keys()):
        v = s["by_type"][t]
        bias = v["mean_signed_miss_all"]
        bias_str = ""
        if v["n"] >= 3:
            if bias > 0:
                bias_str = f" (you tend to UNDER-estimate by {bias:+.3f} on average)"
            elif bias < 0:
                bias_str = f" (you tend to OVER-estimate by {bias:+.3f} on average)"
        lines.append(
            f"  - {t}: {int(v['hit_rate']*v['n'])}/{v['n']} hit rate "
            f"({v['hit_rate']:.0%}){bias_str}"
        )
    if s["n_pending"]:
        lines.append(f"  - {s['n_pending']} predictions pending (not yet resolved)")

    recent = store.recent_resolved(max_recent)
    if recent:
        lines.append("")
        lines.append(f"Most recent resolved predictions (showing up to {max_recent}):")
        for r in recent:
            tag = "HIT" if r.get("in_range") else "MISS"
            key = r.get("key")
            key_str = "×".join(key) if isinstance(key, list) else str(key)
            lines.append(
                f"  [{tag}] {r['type']}({key_str}): predicted "
                f"[{r['low']:+.3f}, {r['high']:+.3f}], actual {r.get('actual', 0):+.3f}"
            )
    return "\n".join(lines)
