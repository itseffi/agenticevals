from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agenticevals.agents.base import AdapterUnavailable, AgentRun, BaseAgent
from agenticevals.model_io import openai_response
from agenticevals.schema import TaskSpec
from agenticevals.tool_parsers import parse_tool_calls
from agenticevals.trace import Trajectory


class ModelLoopAgent(BaseAgent):
    name = "model-loop"

    def run(self, task: TaskSpec, workspace: Path, trace: Trajectory, timeout: int, computer=None) -> AgentRun:
        provider = task.agent.model or os.environ.get("AGENTICEVALS_MODEL_LOOP_PROVIDER", "fixture")
        trace.add("agent.start", agent=self.name, provider=provider, max_steps=task.limits.max_steps)
        messages = [{"role": "user", "content": task.prompt}]
        final = ""
        steps = 0
        for steps in range(1, task.limits.max_steps + 1):
            assistant_text = _next_assistant_text(provider, task, messages, timeout)
            trace.add("agent.model.response", step=steps, chars=len(assistant_text), text=assistant_text[-4000:])
            calls = parse_tool_calls(assistant_text)
            if not calls:
                final = assistant_text.strip()
                trace.add("agent.finish", agent=self.name, steps=steps, final_message=final)
                return AgentRun(ok=True, final_message=final, metadata={"steps": steps, "provider": provider})
            observations = []
            for call in calls:
                trace.add("agent.tool_call.parsed", step=steps, tool_name=call.tool_name, arguments=call.arguments)
                observation = _execute_call(call.tool_name, call.arguments, computer)
                observations.append({"tool_name": call.tool_name, "observation": observation})
                trace.add("agent.tool_call.observation", step=steps, tool_name=call.tool_name, observation=observation)
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "tool", "content": json.dumps(observations, sort_keys=True)})
        return AgentRun(ok=False, final_message=f"model loop exceeded max_steps={task.limits.max_steps}", metadata={"steps": steps})


def _next_assistant_text(provider: str, task: TaskSpec, messages: list[dict[str, str]], timeout: int) -> str:
    if provider == "fixture":
        index = sum(1 for message in messages if message["role"] == "assistant")
        if index >= len(task.agent.script):
            return "Done."
        step = task.agent.script[index]
        return str(step.get("content") or step.get("message") or "")
    if provider == "openai":
        return _openai_text(task, messages, timeout)
    raise AdapterUnavailable(f"model-loop provider is not supported: {provider}")


def _openai_text(task: TaskSpec, messages: list[dict[str, str]], timeout: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("AGENTICEVALS_DEFAULT_MODEL") or task.agent.model
    if not api_key or not model:
        raise AdapterUnavailable("model-loop openai requires OPENAI_API_KEY and a model")
    input_text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return openai_response(model, input_text, timeout=timeout).text


def _execute_call(tool_name: str, arguments: dict[str, Any], computer) -> dict[str, Any]:
    if computer is None:
        return {"ok": False, "error": "no computer interface"}
    if tool_name in {"terminal", "shell", "exec"}:
        result = computer.terminal(str(arguments.get("command", "")), timeout=int(arguments.get("timeout", 60)))
        return {"ok": result.ok, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if tool_name in {"read_file", "read"}:
        return {"ok": True, "content": computer.read_file(str(arguments["path"]))}
    if tool_name in {"write_file", "write"}:
        computer.write_file(str(arguments["path"]), str(arguments.get("content", "")))
        return {"ok": True}
    if hasattr(computer, "dispatch_tool"):
        result = computer.dispatch_tool(tool_name, arguments)
        return result.to_dict()
    return {"ok": False, "error": f"unknown tool: {tool_name}"}
