"""Persistent research journal — cross-session memory for the agent.

Layout under `data/research/`:
    best.json           current best net-Sharpe run + config
    runs.jsonl          append-only record of every backtest (one JSON per line)
    features/
        <name>.py       source code of every agent-proposed feature
        <name>.json     metadata: description, added_at, revision_count,
                        best_sharpe, best_run_id

On session init, every persisted feature is reloaded through the same
`sandbox.exec_feature` validator that gated its original proposal, so a
hand-edited `features/*.py` still cannot smuggle in imports or I/O.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .io_utils import data_dir
from .sandbox import UnsafeCodeError, exec_feature


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LoadedFeature:
    name: str
    fn: Callable
    source: str
    meta: dict


@dataclass
class Journal:
    root: Path
    load_warnings: list[str] = field(default_factory=list)

    @classmethod
    def default(cls) -> "Journal":
        return cls(root=data_dir() / "research")

    # ----- lifecycle ------------------------------------------------------

    def _ensure(self) -> None:
        (self.root / "features").mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.root.exists()

    def clear(self) -> None:
        import shutil

        if self.root.exists():
            shutil.rmtree(self.root)

    # ----- features -------------------------------------------------------

    def _feature_paths(self, name: str) -> tuple[Path, Path]:
        base = self.root / "features" / name
        return base.with_suffix(".py"), base.with_suffix(".json")

    def save_feature(self, name: str, source: str, description: str) -> dict:
        """Write a feature's source + metadata. Bumps revision_count on overwrite."""
        self._ensure()
        py_path, meta_path = self._feature_paths(name)

        prior = {}
        if meta_path.exists():
            try:
                prior = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                prior = {}

        meta = {
            "name": name,
            "description": description,
            "added_at": prior.get("added_at", _now_iso()),
            "updated_at": _now_iso(),
            "revision_count": int(prior.get("revision_count", 0)) + (1 if prior else 0),
            "best_sharpe": prior.get("best_sharpe"),
            "best_run_id": prior.get("best_run_id"),
        }
        py_path.write_text(source)
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    def load_features(self) -> list[LoadedFeature]:
        """Load every persisted feature. Bad files are skipped and recorded in `load_warnings`."""
        self.load_warnings.clear()
        out: list[LoadedFeature] = []
        feat_dir = self.root / "features"
        if not feat_dir.exists():
            return out

        for py_path in sorted(feat_dir.glob("*.py")):
            name = py_path.stem
            source = py_path.read_text()
            meta_path = py_path.with_suffix(".json")
            meta: dict = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except json.JSONDecodeError:
                    self.load_warnings.append(f"{name}: metadata JSON invalid, treating as empty")

            try:
                fn = exec_feature(source, expected_name=name)
            except UnsafeCodeError as e:
                self.load_warnings.append(f"{name}: rejected by sandbox on reload ({e})")
                continue
            except Exception as e:  # noqa: BLE001
                self.load_warnings.append(f"{name}: reload raised {type(e).__name__}: {e}")
                continue

            out.append(LoadedFeature(name=name, fn=fn, source=source, meta=meta))
        return out

    def update_feature_best(self, name: str, run_id: str, sharpe: float) -> None:
        """Update `best_sharpe`/`best_run_id` for a feature if this run beats the stored value."""
        _py, meta_path = self._feature_paths(name)
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            return
        prior = meta.get("best_sharpe")
        if prior is None or sharpe > prior:
            meta["best_sharpe"] = float(sharpe)
            meta["best_run_id"] = run_id
            meta_path.write_text(json.dumps(meta, indent=2))

    def feature_metadata(self, name: str) -> dict | None:
        _py, meta_path = self._feature_paths(name)
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            return None

    def all_feature_metadata(self) -> list[dict]:
        feat_dir = self.root / "features"
        if not feat_dir.exists():
            return []
        out = []
        for meta_path in sorted(feat_dir.glob("*.json")):
            try:
                out.append(json.loads(meta_path.read_text()))
            except json.JSONDecodeError:
                pass
        return out

    # ----- runs + best ----------------------------------------------------

    def record_run(self, config: dict, summary: dict) -> str:
        """Append a run to runs.jsonl, return the assigned run_id."""
        self._ensure()
        run_id = uuid.uuid4().hex[:12]
        record = {
            "run_id": run_id,
            "timestamp": _now_iso(),
            "config": config,
            "summary": _jsonable(summary),
        }
        with open(self.root / "runs.jsonl", "a") as f:
            f.write(json.dumps(record, default=_default_json) + "\n")

        sharpe = summary.get("sharpe")
        if sharpe is not None:
            # Bump per-feature best if applicable.
            for fname in config.get("feature_weights", {}):
                self.update_feature_best(fname, run_id, sharpe)
            # Update global best.
            best = self.best()
            if best is None or sharpe > best.get("summary", {}).get("sharpe", float("-inf")):
                self._write_best(record)
        return run_id

    def best(self) -> dict | None:
        p = self.root / "best.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def _write_best(self, record: dict) -> None:
        self._ensure()
        (self.root / "best.json").write_text(json.dumps(record, indent=2, default=_default_json))

    def all_runs(self) -> list[dict]:
        p = self.root / "runs.jsonl"
        if not p.exists():
            return []
        out = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def top_runs(self, k: int = 5) -> list[dict]:
        runs = [r for r in self.all_runs() if r.get("summary", {}).get("sharpe") is not None]
        runs.sort(key=lambda r: r["summary"]["sharpe"], reverse=True)
        return runs[:k]

    def recent_runs(self, k: int = 5) -> list[dict]:
        return self.all_runs()[-k:]

    def total_runs(self) -> int:
        p = self.root / "runs.jsonl"
        if not p.exists():
            return 0
        with open(p) as f:
            return sum(1 for line in f if line.strip())


# ----- helpers --------------------------------------------------------------


def _default_json(x):
    try:
        import numpy as np

        if isinstance(x, (np.floating, np.integer)):
            return x.item()
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return str(x)


def _jsonable(d: dict) -> dict:
    return json.loads(json.dumps(d, default=_default_json))


def build_state_recap(journal: Journal, max_features: int = 10, max_top_runs: int = 5) -> str:
    """Compose the `prior research state` section that gets prepended to the user goal.

    Kept compact — this is context the agent re-reads every run.
    """
    lines: list[str] = ["# Prior research state"]
    best = journal.best()
    if best is None:
        lines.append(
            "No previous runs recorded. You are starting from the baseline described in the system prompt."
        )
    else:
        cfg = best.get("config", {})
        s = best.get("summary", {})

        def _fmt(v, kind: str) -> str:
            if v is None:
                return "—"
            if kind == "pct":
                return f"{v:+.2%}"
            return f"{v:+.3f}"

        lines.append(
            f"Current best (run {best.get('run_id')}, {best.get('timestamp')}):"
        )
        lines.append(f"  config: {json.dumps(cfg, default=str)}")
        lines.append(
            "  metrics: "
            f"net sharpe {_fmt(s.get('sharpe'), 'float')}, "
            f"gross sharpe {_fmt(s.get('gross_sharpe'), 'float')}, "
            f"ic IR {_fmt(s.get('ic_ir'), 'float')}, "
            f"turnover {_fmt(s.get('avg_turnover'), 'pct')}, "
            f"max DD {_fmt(s.get('max_drawdown'), 'pct')}."
        )

    feats = journal.all_feature_metadata()
    if feats:
        lines.append("")
        lines.append(
            f"Previously-proposed features (already in registry, {len(feats)} total, showing up to {max_features}):"
        )
        for m in feats[:max_features]:
            best_s = m.get("best_sharpe")
            best_str = f"best net sharpe seen: {best_s:+.3f}" if best_s is not None else "not yet backtested in this config"
            desc = (m.get("description") or "").strip().replace("\n", " ")
            if len(desc) > 120:
                desc = desc[:117] + "..."
            lines.append(f"  - {m['name']}: {desc} ({best_str})")

    top = journal.top_runs(max_top_runs)
    if len(top) >= 2:
        lines.append("")
        lines.append(f"Top {len(top)} runs on record by net sharpe:")
        for r in top:
            s = r.get("summary", {})
            c = r.get("config", {})
            lines.append(
                f"  - {r['run_id']}: sharpe {s.get('sharpe', 0):+.3f}, "
                f"ic_ir {s.get('ic_ir', 0):+.3f}, "
                f"weights={c.get('feature_weights')}, "
                f"hl={c.get('halflife_days')}, weighting={c.get('weighting')}"
            )

    total = journal.total_runs()
    lines.append("")
    lines.append(f"Total backtests on record: {total}.")
    lines.append(
        "Build on this — don't re-propose features already in the registry unless "
        "you have a distinctly new variant, and explain why a previously-weak feature "
        "should work now."
    )

    # Calibration track record (if populated).
    try:
        from .calibration import CalibrationStore, build_calibration_recap

        cal_recap = build_calibration_recap(CalibrationStore.default())
        if cal_recap:
            lines.append("")
            lines.append(cal_recap)
    except Exception:  # noqa: BLE001 — defensive; missing store shouldn't kill the recap
        pass

    return "\n".join(lines)
