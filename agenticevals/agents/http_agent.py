from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from agenticevals.agents.base import AgentRun, BaseAgent
from agenticevals.schema import TaskSpec
from agenticevals.trace import Trajectory


class HTTPAgent(BaseAgent):
    name = "http"

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def run(self, task: TaskSpec, workspace: Path, trace: Trajectory, timeout: int, computer=None) -> AgentRun:
        payload = {
            "task": {
                "id": task.id,
                "title": task.title,
                "prompt": task.prompt,
                "tools": [{"name": tool.name, "description": tool.description, "input_schema": tool.input_schema} for tool in task.tools],
            },
            "workspace": str(workspace),
            "sandbox_url": getattr(computer, "sandbox_url", None),
            "limits": {"max_steps": task.limits.max_steps, "max_minutes": task.limits.max_minutes},
        }
        trace.add("agent.start", agent=self.name, endpoint=self.endpoint, sandbox_url=payload["sandbox_url"])
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        ok = bool(body.get("ok", response.status < 400))
        final = str(body.get("final_message") or body.get("message") or "")
        for event in body.get("events", []):
            if isinstance(event, dict):
                trace.add("agent.trace", source="http", event=event)
        trace.add("agent.finish", agent=self.name, endpoint=self.endpoint, status=response.status, ok=ok)
        return AgentRun(ok=ok, final_message=final, metadata={"status": response.status, "endpoint": self.endpoint})


def endpoint_from_env_or_task(task: TaskSpec) -> str:
    endpoint = task.agent.command or os.environ.get("AGENTICEVALS_HTTP_AGENT_URL")
    if not endpoint:
        raise ValueError("http agent requires task.agent.command or AGENTICEVALS_HTTP_AGENT_URL")
    return endpoint
