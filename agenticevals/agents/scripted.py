from __future__ import annotations

from pathlib import Path

from agenticevals.agents.base import AgentRun, BaseAgent
from agenticevals.schema import TaskSpec
from agenticevals.trace import Trajectory
from agenticevals.utils import safe_relative_path


class ScriptedAgent(BaseAgent):
    name = "scripted"

    def run(self, task: TaskSpec, workspace: Path, trace: Trajectory, timeout: int, computer=None) -> AgentRun:
        if len(task.agent.script) > task.limits.max_steps:
            return AgentRun(
                ok=False,
                final_message=f"scripted agent has {len(task.agent.script)} steps; limit is {task.limits.max_steps}",
                metadata={"steps": len(task.agent.script), "max_steps": task.limits.max_steps},
            )
        trace.add("agent.start", agent=self.name, steps=len(task.agent.script))
        final_message = "scripted agent completed"
        for step_index, step in enumerate(task.agent.script, start=1):
            action = step.get("action")
            trace.add("agent.step", index=step_index, action=action, data=step)
            if action == "write_file":
                rel = safe_relative_path(str(step["path"]))
                if computer is not None:
                    computer.write_file(str(rel), str(step.get("content", "")))
                else:
                    path = workspace / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(str(step.get("content", "")), encoding="utf-8")
                trace.add("computer.file.write", path=str(rel))
            elif action == "replace":
                rel = safe_relative_path(str(step["path"]))
                text = computer.read_file(str(rel)) if computer is not None else (workspace / rel).read_text(encoding="utf-8")
                old = str(step["old"])
                new = str(step["new"])
                if old not in text:
                    trace.add("agent.error", message=f"replacement text not found in {rel}")
                    return AgentRun(ok=False, final_message=f"replacement text not found in {rel}", metadata={})
                if computer is not None:
                    computer.write_file(str(rel), text.replace(old, new, 1))
                else:
                    (workspace / rel).write_text(text.replace(old, new, 1), encoding="utf-8")
                trace.add("computer.file.replace", path=str(rel))
            elif action == "run":
                if computer is not None:
                    result = computer.terminal(str(step["command"]), timeout=timeout)
                else:
                    from agenticevals.computer.shell import Shell

                    result = Shell(workspace, trace).run(str(step["command"]), timeout=timeout, event_type="agent.shell")
                if step.get("must_pass", False) and not result.ok:
                    return AgentRun(ok=False, final_message=f"command failed: {step['command']}", metadata={})
            elif action == "tool_call":
                if computer is None or not hasattr(computer, "dispatch_tool"):
                    return AgentRun(ok=False, final_message="scripted tool_call requires a tool dispatcher", metadata={})
                result = computer.dispatch_tool(str(step["tool_name"]), dict(step.get("arguments", {})))
                if step.get("must_pass", True) and not result.ok:
                    return AgentRun(ok=False, final_message=f"tool call failed: {step['tool_name']}", metadata={"error": result.error})
            elif action == "final":
                final_message = str(step.get("message", final_message))
            else:
                return AgentRun(ok=False, final_message=f"unknown scripted action: {action}", metadata={})
        trace.add("agent.finish", agent=self.name, final_message=final_message)
        return AgentRun(ok=True, final_message=final_message, metadata={"steps": len(task.agent.script)})
