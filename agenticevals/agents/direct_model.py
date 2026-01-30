from __future__ import annotations

import os
from pathlib import Path

from agenticevals.agents.base import AdapterUnavailable, AgentRun, BaseAgent
from agenticevals.model_io import openai_response
from agenticevals.schema import TaskSpec
from agenticevals.trace import Trajectory


class DirectModelAgent(BaseAgent):
    name = "direct-model"

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def run(self, task: TaskSpec, workspace: Path, trace: Trajectory, timeout: int, computer=None) -> AgentRun:
        trace.add("agent.start", agent=self.name, provider=self.provider, model=self.model)
        if self.provider == "openai":
            return self._run_openai(task, timeout, trace)
        raise AdapterUnavailable(f"direct-model provider is not supported: {self.provider}")

    def _run_openai(self, task: TaskSpec, timeout: int, trace: Trajectory) -> AgentRun:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise AdapterUnavailable("direct-model requires OPENAI_API_KEY")
        if not self.model:
            raise AdapterUnavailable("direct-model requires AGENTICEVALS_DEFAULT_MODEL or agent.model")
        try:
            response = openai_response(
                self.model,
                (
                    "You are being evaluated as an AI agent. Return a concise plan and final answer. "
                    "This direct adapter does not grant filesystem tools.\n\n"
                    f"Task:\n{task.prompt}"
                ),
                timeout=timeout,
            )
        except Exception as exc:
            raise AdapterUnavailable(f"direct-model request failed: {exc}") from exc
        trace.add("agent.finish", agent=self.name, output_chars=len(response.text), cached=response.cached, cost_usd=response.cost_usd, usage=response.usage)
        return AgentRun(ok=True, final_message=response.text, metadata={"provider": self.provider, "model": self.model, "cached": response.cached, "cost_usd": response.cost_usd, "usage": response.usage})
