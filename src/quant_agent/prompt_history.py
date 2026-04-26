"""Versioned prompt store for the meta-agent.

Each agent's system prompt is treated as a versioned artifact. When the
meta-agent proposes a rewrite, it produces a new version with a parent
pointer. Promotion (or rejection) is recorded against the version. This
lets the meta-agent reason about which prompt versions performed which
in-sample / OOS metrics, and roll back if a rewrite proves harmful.

Storage: data/research/prompts/{agent_name}/v{N}.json
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .io_utils import data_dir


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PromptVersion:
    agent: str
    version: int
    parent_version: int | None
    text: str
    rationale: str
    created_at: str
    promoted: bool = False
    promoted_at: str | None = None
    metrics_at_promotion: dict = field(default_factory=dict)
    rolled_back: bool = False
    rolled_back_at: str | None = None
    text_hash: str = ""

    @classmethod
    def new(
        cls,
        agent: str,
        version: int,
        parent_version: int | None,
        text: str,
        rationale: str,
    ) -> "PromptVersion":
        return cls(
            agent=agent,
            version=version,
            parent_version=parent_version,
            text=text,
            rationale=rationale,
            created_at=_now_iso(),
            text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        )

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "version": self.version,
            "parent_version": self.parent_version,
            "text": self.text,
            "rationale": self.rationale,
            "created_at": self.created_at,
            "promoted": self.promoted,
            "promoted_at": self.promoted_at,
            "metrics_at_promotion": self.metrics_at_promotion,
            "rolled_back": self.rolled_back,
            "rolled_back_at": self.rolled_back_at,
            "text_hash": self.text_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PromptVersion":
        return cls(
            agent=d["agent"],
            version=int(d["version"]),
            parent_version=(int(d["parent_version"]) if d.get("parent_version") is not None else None),
            text=d["text"],
            rationale=d.get("rationale", ""),
            created_at=d.get("created_at", _now_iso()),
            promoted=bool(d.get("promoted", False)),
            promoted_at=d.get("promoted_at"),
            metrics_at_promotion=dict(d.get("metrics_at_promotion") or {}),
            rolled_back=bool(d.get("rolled_back", False)),
            rolled_back_at=d.get("rolled_back_at"),
            text_hash=d.get("text_hash", ""),
        )


@dataclass
class PromptHistory:
    root: Path

    @classmethod
    def default(cls) -> "PromptHistory":
        return cls(root=data_dir() / "research" / "prompts")

    def _agent_dir(self, agent: str) -> Path:
        p = self.root / agent
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _version_path(self, agent: str, version: int) -> Path:
        return self._agent_dir(agent) / f"v{version:04d}.json"

    def list_versions(self, agent: str) -> list[PromptVersion]:
        d = self._agent_dir(agent)
        out: list[PromptVersion] = []
        for path in sorted(d.glob("v*.json")):
            try:
                out.append(PromptVersion.from_dict(json.loads(path.read_text())))
            except json.JSONDecodeError:
                continue
        return out

    def latest(self, agent: str) -> PromptVersion | None:
        versions = self.list_versions(agent)
        if not versions:
            return None
        return max(versions, key=lambda v: v.version)

    def latest_promoted(self, agent: str) -> PromptVersion | None:
        promoted = [v for v in self.list_versions(agent) if v.promoted and not v.rolled_back]
        if not promoted:
            return None
        return max(promoted, key=lambda v: v.version)

    def add_proposal(
        self,
        agent: str,
        text: str,
        rationale: str,
        parent: PromptVersion | None = None,
    ) -> PromptVersion:
        """Register a new proposed prompt version (not yet promoted)."""
        existing = self.list_versions(agent)
        next_v = (max((v.version for v in existing), default=0)) + 1
        parent_v = parent.version if parent else (existing[-1].version if existing else None)
        pv = PromptVersion.new(agent=agent, version=next_v, parent_version=parent_v, text=text, rationale=rationale)
        self._version_path(agent, next_v).write_text(json.dumps(pv.to_dict(), indent=2, default=str))
        return pv

    def promote(self, pv: PromptVersion, metrics: dict) -> None:
        pv.promoted = True
        pv.promoted_at = _now_iso()
        pv.metrics_at_promotion = dict(metrics)
        self._version_path(pv.agent, pv.version).write_text(
            json.dumps(pv.to_dict(), indent=2, default=str)
        )

    def rollback(self, pv: PromptVersion) -> None:
        pv.rolled_back = True
        pv.rolled_back_at = _now_iso()
        self._version_path(pv.agent, pv.version).write_text(
            json.dumps(pv.to_dict(), indent=2, default=str)
        )

    def diff(self, agent: str, v_from: int, v_to: int) -> str:
        """Return a unified-style diff between two versions."""
        try:
            a = PromptVersion.from_dict(json.loads(self._version_path(agent, v_from).read_text()))
            b = PromptVersion.from_dict(json.loads(self._version_path(agent, v_to).read_text()))
        except FileNotFoundError as e:
            return f"diff failed: {e}"
        a_lines = a.text.splitlines()
        b_lines = b.text.splitlines()
        # Tiny line-level diff (no library, keep it portable)
        out = [f"--- {agent}/v{v_from:04d}", f"+++ {agent}/v{v_to:04d}"]
        i = j = 0
        while i < len(a_lines) and j < len(b_lines):
            if a_lines[i] == b_lines[j]:
                i += 1
                j += 1
            else:
                out.append(f"- {a_lines[i]}")
                out.append(f"+ {b_lines[j]}")
                i += 1
                j += 1
        for line in a_lines[i:]:
            out.append(f"- {line}")
        for line in b_lines[j:]:
            out.append(f"+ {line}")
        return "\n".join(out)


# ----- helpers --------------------------------------------------------------


def slugify(text: str) -> str:
    return re.sub(r"\W+", "_", text.lower())[:64]
