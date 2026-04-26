"""Multi-agent specs for the quant-agent orchestrator.

Each agent is defined as an `AgentSpec` — its system prompt, the tools it
can call, and the goal-template it expects from the orchestrator. Specs
are pure data; the orchestrator wires them to an Anthropic client at run
time. This separation makes specs unit-testable without API access.
"""
from .base import AgentSpec, AgentResult
from .alpha import alpha_agent_spec
from .portfolio import portfolio_agent_spec
from .cost_risk import cost_risk_agent_spec
from .critic import critic_agent_spec
from .meta import meta_agent_spec

__all__ = [
    "AgentSpec",
    "AgentResult",
    "alpha_agent_spec",
    "portfolio_agent_spec",
    "cost_risk_agent_spec",
    "critic_agent_spec",
    "meta_agent_spec",
]
