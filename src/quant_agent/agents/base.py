"""Base classes for agent specifications.

An `AgentSpec` is the data contract between the orchestrator and a Claude
agent. It carries:
  * a name (for logging + journaling)
  * a system prompt template (rendered against an IPS at run time)
  * a tool set (a list of @beta_tool callables; built lazily by a factory
    so they can close over the live ResearchSession + IPS)
  * a model + thinking + effort policy
  * a goal-prompt template — what the orchestrator hands the agent

Specs are pure data. The orchestrator owns API calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..ips import IPS


@dataclass
class AgentResult:
    """Standard return shape from any agent invocation.

    `outputs` is the agent's structured findings (weights, signal, summary, etc.)
    `transcript` is the message-level record for inspection.
    `usage` is token consumption (for cost accounting).
    """
    agent_name: str
    success: bool
    outputs: dict
    transcript: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class AgentSpec:
    """The static configuration of an agent role.

    `system_prompt_fn` and `tool_factory` are callables that take an IPS
    + (optional) session-like context and return the rendered prompt + the
    list of @beta_tool callables. This keeps the spec independent of any
    particular ResearchSession instance — orchestrator binds them at run.
    """
    name: str
    description: str
    system_prompt_fn: Callable[..., str]
    tool_factory: Callable[..., list]
    model: str = "claude-opus-4-7"
    use_thinking: bool = True
    effort: str = "high"
    max_tokens: int = 16000
    max_iterations: int = 12

    def render_system_prompt(self, ips: IPS, **ctx) -> str:
        return self.system_prompt_fn(ips, **ctx)

    def build_tools(self, **ctx) -> list:
        return self.tool_factory(**ctx)
