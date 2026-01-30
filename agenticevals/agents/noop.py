from __future__ import annotations

from pathlib import Path

from agenticevals.agents.base import AgentRun, BaseAgent
from agenticevals.schema import TaskSpec
from agenticevals.trace import Trajectory


class NoopAgent(BaseAgent):
    name = "noop"

    def run(self, task: TaskSpec, workspace: Path, trace: Trajectory, timeout: int, computer=None) -> AgentRun:
        trace.add("agent.start", agent=self.name)
        trace.add("agent.finish", agent=self.name, final_message="noop agent made no changes")
        return AgentRun(ok=True, final_message="noop agent made no changes", metadata={"steps": 0})
