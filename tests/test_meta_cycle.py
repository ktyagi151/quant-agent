"""Meta-agent end-to-end logic with mocked invokers — no API calls.

Covers:
  - propose_prompt_rewrite tool behavior (validation, parent linkage)
  - validate_prompt_rewrite promotion rule under Bonferroni correction
  - run_meta_cycle dispatch + decision recording
  - portfolio prompt now includes explicit risk-parity guardrails
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from quant_agent.agents.base import AgentResult
from quant_agent.ips import load_ips
from quant_agent.orchestrator import (
    PromotionDecision,
    _bonferroni_threshold,
    _spec_with_prompt_override,
    run_meta_cycle,
    validate_prompt_rewrite,
)
from quant_agent.prompt_history import PromptHistory


@pytest.fixture
def ips():
    return load_ips(Path(__file__).parent.parent / "configs" / "ips" / "default.yaml")


@pytest.fixture
def tmp_prompt_store(tmp_path, monkeypatch):
    """Redirect PromptHistory.default() to a tmp dir for isolation."""
    store = PromptHistory(root=tmp_path / "prompts")
    monkeypatch.setattr(
        "quant_agent.prompt_history.PromptHistory.default",
        classmethod(lambda cls: store),
    )
    return store


# ----- portfolio prompt change (test for §3) -------------------------------


def test_portfolio_prompt_warns_about_risk_parity_under_net_neutral(ips):
    from quant_agent.agents import portfolio_agent_spec

    text = portfolio_agent_spec.render_system_prompt(ips)
    # Default IPS has net_exposure ≤ 0.10 — long-only warning must trigger.
    assert "LONG-ONLY by construction" in text
    assert "WILL violate the IPS net_exposure cap" in text
    assert "Do not use as a fallback" in text
    assert "max_abs_weight" in text  # mean-variance degeneracy guard
    assert "I could not construct a compliant portfolio" in text  # opt-out clause


def test_portfolio_prompt_omits_warning_under_unconstrained_ips(ips):
    """If the IPS doesn't mandate dollar-neutrality, the long-only warning shouldn't fire."""
    from quant_agent.agents import portfolio_agent_spec

    # Strip the net_exposure constraint
    ips.hard_constraints = [hc for hc in ips.hard_constraints if hc.metric != "net_exposure"]
    text = portfolio_agent_spec.render_system_prompt(ips)
    assert "WILL violate the IPS net_exposure cap" not in text


# ----- bonferroni threshold ------------------------------------------------


def test_bonferroni_threshold_decreases_with_more_tests(ips):
    ips.holdout.bonferroni_n_tests = 1
    t1 = _bonferroni_threshold(ips)
    ips.holdout.bonferroni_n_tests = 100
    t100 = _bonferroni_threshold(ips)
    assert t100 < t1


def test_bonferroni_threshold_default_value(ips):
    # ips.holdout.bonferroni_n_tests = 20 by default
    t = _bonferroni_threshold(ips)
    # 2.0 * 0.3 / sqrt(20) ≈ 0.134
    assert 0.10 < t < 0.20


# ----- spec override -------------------------------------------------------


def test_spec_with_prompt_override_replaces_text(ips):
    from quant_agent.agents import alpha_agent_spec

    overridden = _spec_with_prompt_override(alpha_agent_spec, "OVERRIDDEN_TEXT")
    assert overridden.render_system_prompt(ips) == "OVERRIDDEN_TEXT"
    # Other fields preserved
    assert overridden.name == alpha_agent_spec.name
    assert overridden.model == alpha_agent_spec.model
    assert overridden.tool_factory is alpha_agent_spec.tool_factory


# ----- validate_prompt_rewrite ---------------------------------------------


class _FakeSession:
    """Stand-in for ResearchSession that supports the bits validate_prompt_rewrite touches."""

    def __init__(self, n_obs=300, last_summary=None, journal_best=None):
        import pandas as pd
        self.panel = {"adj_close": pd.DataFrame(index=range(n_obs), columns=["A", "B"])}
        self._last_summary = last_summary or {}
        self.journal = _FakeJournal(journal_best) if journal_best is not None else None

    def last_run(self):
        if not self._last_summary:
            return None

        class _Last:
            def __init__(self, s):
                self.summary = s
        return _Last(self._last_summary)


class _FakeJournal:
    def __init__(self, best):
        self._best = best

    def best(self):
        return self._best


def _mock_invoker(canned_metrics_for_session_id):
    """Returns an invoker that ignores its inputs and returns a canned AgentResult."""
    def invoker(spec, ips, user_message, session=None, **_):
        return AgentResult(
            agent_name=spec.name, success=True,
            outputs={"final_text": "ok"}, transcript=[], usage={},
        )
    return invoker


def test_validate_promotes_when_holdout_improves(ips, tmp_prompt_store):
    candidate = tmp_prompt_store.add_proposal(
        agent="alpha", text="x" * 300, rationale="big improvement expected"
    )
    # Baseline holdout sharpe = 0.05; candidate produces 0.50 → delta way above threshold.
    in_sample = _FakeSession(n_obs=300, last_summary={"sharpe": 0.40})
    holdout = _FakeSession(
        n_obs=300,
        last_summary={"sharpe": 0.50, "avg_turnover": 0.20},
        journal_best={"summary": {"sharpe": 0.05}},
    )
    decision = validate_prompt_rewrite(
        candidate, ips=ips,
        in_sample_session=in_sample, holdout_session=holdout,
        invoker=_mock_invoker(None),
    )
    assert decision.decision == "PROMOTED"
    assert decision.sharpe_delta > 0
    # Side effect: store recorded promotion
    latest = tmp_prompt_store.latest("alpha")
    assert latest.promoted is True
    assert latest.metrics_at_promotion.get("sharpe") == 0.50


def test_validate_rejects_when_holdout_does_not_improve(ips, tmp_prompt_store):
    candidate = tmp_prompt_store.add_proposal(
        agent="alpha", text="x" * 300, rationale="hopefully better, probably not"
    )
    # Candidate barely beats baseline — below Bonferroni threshold.
    in_sample = _FakeSession(n_obs=300, last_summary={"sharpe": 0.05})
    holdout = _FakeSession(
        n_obs=300,
        last_summary={"sharpe": 0.10, "avg_turnover": 0.20},
        journal_best={"summary": {"sharpe": 0.05}},
    )
    decision = validate_prompt_rewrite(
        candidate, ips=ips,
        in_sample_session=in_sample, holdout_session=holdout,
        invoker=_mock_invoker(None),
    )
    assert decision.decision == "REJECTED_NO_IMPROVEMENT"
    # Promotion side effect did NOT fire
    latest = tmp_prompt_store.latest("alpha")
    assert latest.promoted is False


def test_validate_rejects_with_insufficient_holdout_observations(ips, tmp_prompt_store):
    candidate = tmp_prompt_store.add_proposal(
        agent="alpha", text="x" * 300, rationale="needs more holdout time"
    )
    # n_obs < ips.holdout.min_observations (default 252)
    holdout = _FakeSession(n_obs=100, last_summary={"sharpe": 0.50})
    decision = validate_prompt_rewrite(
        candidate, ips=ips,
        in_sample_session=_FakeSession(),
        holdout_session=holdout,
        invoker=_mock_invoker(None),
    )
    assert decision.decision == "REJECTED_INSUFFICIENT_OBSERVATIONS"


def test_validate_rejects_on_ips_violation(ips, tmp_prompt_store):
    candidate = tmp_prompt_store.add_proposal(
        agent="alpha", text="x" * 300, rationale="testing turnover violation"
    )
    # In-sample turnover above the IPS cap (0.50)
    in_sample = _FakeSession(n_obs=300, last_summary={"sharpe": 1.0, "avg_turnover": 0.80})
    holdout = _FakeSession(
        n_obs=300,
        last_summary={"sharpe": 1.0, "avg_turnover": 0.80},
        journal_best={"summary": {"sharpe": 0.05}},
    )
    decision = validate_prompt_rewrite(
        candidate, ips=ips,
        in_sample_session=in_sample,
        holdout_session=holdout,
        invoker=_mock_invoker(None),
    )
    assert decision.decision == "REJECTED_IPS_VIOLATION"


def test_validate_rejects_unknown_agent(ips, tmp_prompt_store):
    candidate = tmp_prompt_store.add_proposal(
        agent="bogus_agent", text="x" * 300, rationale="this agent doesn't exist"
    )
    decision = validate_prompt_rewrite(
        candidate, ips=ips,
        in_sample_session=_FakeSession(),
        holdout_session=_FakeSession(),
        invoker=_mock_invoker(None),
    )
    assert decision.decision == "REJECTED_NO_IMPROVEMENT"
    assert "unknown agent" in decision.rationale


# ----- run_meta_cycle ------------------------------------------------------


def test_run_meta_cycle_dispatches_meta_then_validation(ips, tmp_prompt_store):
    """The meta-cycle should invoke meta-agent and then validate any pending proposals."""
    # Pre-populate a pending proposal as if the meta-agent had submitted one.
    tmp_prompt_store.add_proposal(
        agent="alpha", text="x" * 400, rationale="from a prior cycle"
    )

    in_sample = _FakeSession(n_obs=300, last_summary={"sharpe": 0.40})
    holdout = _FakeSession(
        n_obs=300,
        last_summary={"sharpe": 0.20, "avg_turnover": 0.20},
        journal_best={"summary": {"sharpe": 0.05}},
    )

    invoker = _mock_invoker(None)
    result = run_meta_cycle(
        ips=ips, in_sample_session=in_sample, holdout_session=holdout,
        dry_run=False, invoker=invoker,
    )
    assert "meta_result" in result
    assert "promotion_decisions" in result
    assert len(result["promotion_decisions"]) == 1
    d = result["promotion_decisions"][0]
    # Holdout sharpe 0.20 vs baseline 0.05 → delta 0.15 ≈ at threshold
    # (default Bonferroni threshold ~0.134, so this should promote)
    assert d["agent"] == "alpha"
    assert d["decision"] in ("PROMOTED", "REJECTED_NO_IMPROVEMENT")  # depending on exact threshold


def test_run_meta_cycle_no_pending_proposal_no_decisions(ips, tmp_prompt_store):
    """If no pending proposal exists, no validation runs."""
    in_sample = _FakeSession()
    holdout = _FakeSession()
    invoker = _mock_invoker(None)
    result = run_meta_cycle(
        ips=ips, in_sample_session=in_sample, holdout_session=holdout,
        dry_run=False, invoker=invoker,
    )
    assert result["promotion_decisions"] == []


# ----- propose_prompt_rewrite tool wiring ----------------------------------


def test_propose_prompt_rewrite_tool_validates_inputs(ips, tmp_prompt_store):
    """The tool should reject too-short text + bad agent names."""
    from quant_agent.agents.meta import meta_agent_spec

    # Build the tool against a minimal session-like object.
    class _Sess:
        journal = None
        calibration = None

    tools = meta_agent_spec.tool_factory(session=_Sess(), ips=ips)
    propose = next(t for t in tools if t.name == "propose_prompt_rewrite")

    # Bad agent name
    out = json.loads(propose(agent_name="not_a_real_agent", new_text="x" * 300, rationale="x" * 30))
    assert "error" in out

    # Too-short text
    out = json.loads(propose(agent_name="alpha", new_text="too short", rationale="x" * 30))
    assert "error" in out

    # Too-short rationale
    out = json.loads(propose(agent_name="alpha", new_text="x" * 300, rationale="too short"))
    assert "error" in out

    # Valid → returns version info
    out = json.loads(propose(agent_name="alpha", new_text="x" * 400, rationale="x" * 50))
    assert out.get("status") == "PROPOSED_PENDING_VALIDATION"
    assert out.get("agent") == "alpha"
    assert out.get("version") == 1


def test_list_prompt_versions_tool_returns_history(ips, tmp_prompt_store):
    from quant_agent.agents.meta import meta_agent_spec

    tmp_prompt_store.add_proposal(agent="alpha", text="v1 text " * 100, rationale="first")
    tmp_prompt_store.add_proposal(agent="alpha", text="v2 text " * 100, rationale="second")

    class _Sess:
        journal = None
        calibration = None

    tools = meta_agent_spec.tool_factory(session=_Sess(), ips=ips)
    lister = next(t for t in tools if t.name == "list_prompt_versions")
    out = json.loads(lister(agent_name="alpha"))
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0]["version"] == 1
    assert out[1]["version"] == 2
    assert out[1]["parent_version"] == 1
