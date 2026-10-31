"""The agent loop: ask the brain for an action, run the tool, feed the observation back, repeat."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .brains import Brain, BrainError, Step, make_brains
from .config import Config
from .memory import Memory
from .tools import Tool, ToolContext, ToolError, build_tools
from .workspace import Workspace

log = logging.getLogger("agentforge")
MAX_OBSERVATION = 20_000


@dataclass
class RunResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    brain: str = ""
    degraded: bool = False  # True when the preferred brain failed and a later one in the chain answered


class Agent:
    """Runs the decide -> act -> observe loop over an ordered chain of brains (first healthy one wins)."""

    def __init__(self, brains: list[Brain], tools: dict[str, Tool], max_steps: int = 6, role: str = "Assistant",
                 allowed_tools: frozenset[str] | None = None) -> None:
        if not brains:
            raise ValueError("at least one brain is required")
        self.brains, self.tools, self.max_steps, self.role = brains, tools, max_steps, role
        self.allowed_tools = allowed_tools

    @property
    def brain(self) -> Brain:
        return self.brains[0]

    def with_role(self, role: str, allowed_tools: frozenset[str] | None = None) -> "Agent":
        return Agent(self.brains, self.tools, self.max_steps, role, allowed_tools)

    def _visible_tools(self) -> dict[str, Tool]:
        if self.allowed_tools is None:
            return self.tools
        return {k: t for k, t in self.tools.items() if k in self.allowed_tools}

    def run(self, goal: str, context: str = "") -> RunResult:
        goal = goal.strip()
        if not goal:
            raise ValueError("goal is empty")
        if len(goal) > 20_000:
            raise ValueError("goal is too long")
        errors: list[str] = []
        for index, brain in enumerate(self.brains):
            try:
                result = self._loop(brain, goal, context)
            except BrainError as exc:
                log.warning("brain failed", extra={"brain": brain.name, "error": str(exc)})
                errors.append(f"{brain.name}: {exc}")
                continue
            result.degraded = index > 0
            return result
        return RunResult("All brains failed: " + "; ".join(errors), brain=self.brains[-1].name)

    def _loop(self, brain: Brain, goal: str, context: str) -> RunResult:
        tools = self._visible_tools()
        history: list[Step] = []
        for _ in range(self.max_steps):
            action = brain.decide(goal, context, history, tools, self.role)
            if action.final is not None:
                return RunResult(action.final, history, brain.name)
            tool = tools.get(action.tool or "")
            if tool is None:
                history.append(Step(action.tool or "?", action.args, "Tool error: tool not available to this role", ok=False))
                continue
            try:
                obs, ok = tool.call(action.args)[:MAX_OBSERVATION], True
            except ToolError as exc:
                obs, ok = f"Tool error: {exc}", False
            history.append(Step(tool.name, action.args, obs, ok))
        last = history[-1].observation if history else "No answer produced."
        return RunResult(last, history, brain.name)


@dataclass
class Runtime:
    config: Config
    memory: Memory
    workspace: Workspace
    agent: Agent


def build_runtime(config: Config) -> Runtime:
    config.validate()
    memory = Memory(config.memory_path)
    workspace = Workspace(config.workspace_dir)
    tools = build_tools(ToolContext(memory, workspace, config))
    agent = Agent(make_brains(config), tools, config.max_steps)
    return Runtime(config, memory, workspace, agent)
