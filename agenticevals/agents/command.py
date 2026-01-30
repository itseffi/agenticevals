from __future__ import annotations

import shlex
import os
import json
from pathlib import Path

from agenticevals.agents.base import AgentRun, BaseAgent
from agenticevals.schema import TaskSpec
from agenticevals.trace import Trajectory
from agenticevals.trace_ingest import ingest_jsonl_trace
from agenticevals.utils import run_command


class CommandAgent(BaseAgent):
    name = "command"

    def __init__(self, command_template: str, name: str = "command"):
        self.command_template = command_template
        self.name = name

    def run(self, task: TaskSpec, workspace: Path, trace: Trajectory, timeout: int, computer=None) -> AgentRun:
        control_dir = workspace / ".agenticevals"
        control_dir.mkdir(exist_ok=True)
        prompt_path = control_dir / "prompt.txt"
        task_path = control_dir / "task.json"
        result_path = control_dir / "result.json"
        trace_path = control_dir / "agent-trace.jsonl"
        prompt_text = _agent_prompt(task, workspace, computer)
        prompt_path.write_text(prompt_text, encoding="utf-8")
        task_path.write_text(json.dumps(_task_packet(task, workspace, computer), indent=2, sort_keys=True), encoding="utf-8")
        command = self.command_template.format(
            prompt=shlex.quote(prompt_text),
            prompt_path=shlex.quote(str(prompt_path)),
            task_path=shlex.quote(str(task_path)),
            result_path=shlex.quote(str(result_path)),
            trace_path=shlex.quote(str(trace_path)),
            workspace=shlex.quote(str(workspace)),
        )
        env = dict(os.environ)
        env.update(
            {
                "AGENTICEVALS_AGENT_NAME": self.name,
                "AGENTICEVALS_ROOT": str(Path(__file__).resolve().parents[2]),
                "AGENTICEVALS_WORKSPACE": str(workspace),
                "AGENTICEVALS_PROMPT_PATH": str(prompt_path),
                "AGENTICEVALS_TASK_PATH": str(task_path),
                "AGENTICEVALS_RESULT_PATH": str(result_path),
                "AGENTICEVALS_TRACE_PATH": str(trace_path),
            }
        )
        sandbox_url = getattr(computer, "sandbox_url", None)
        if sandbox_url:
            env["AGENTICEVALS_SANDBOX_URL"] = str(sandbox_url)
        trace.add(
            "agent.start",
            agent=self.name,
            command=command,
            prompt_path=str(prompt_path),
            task_path=str(task_path),
            result_path=str(result_path),
            trace_path=str(trace_path),
            sandbox_url=sandbox_url,
        )
        result = run_command(command, workspace, timeout=timeout, env=env)
        ingested = ingest_jsonl_trace(trace_path, trace, source=self.name)
        declared_result = _load_declared_result(result_path, trace)
        final_message = declared_result.get("final_message") if declared_result else None
        ok = declared_result.get("ok") if declared_result and isinstance(declared_result.get("ok"), bool) else result.ok
        trace.add(
            "agent.finish",
            agent=self.name,
            returncode=result.returncode,
            stdout=result.stdout[-8000:],
            stderr=result.stderr[-8000:],
            timed_out=result.timed_out,
            ingested_trace_events=ingested,
            declared_result=declared_result,
        )
        return AgentRun(
            ok=bool(ok),
            final_message=str(final_message or result.stdout.strip() or result.stderr.strip()),
            metadata={"returncode": result.returncode, "timed_out": result.timed_out, "ingested_trace_events": ingested},
        )


def _task_packet(task: TaskSpec, workspace: Path, computer) -> dict:
    return {
        "id": task.id,
        "title": task.title,
        "prompt": task.prompt,
        "workspace": str(workspace),
        "limits": {"max_steps": task.limits.max_steps, "max_minutes": task.limits.max_minutes},
        "tools": [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in task.tools
        ],
        "tool_endpoints": [
            {"tool_name": endpoint.tool_name, "url": endpoint.url, "method": endpoint.method}
            for endpoint in task.tool_endpoints
        ],
        "sandbox": {"url": getattr(computer, "sandbox_url", None)},
        "result_path": ".agenticevals/result.json",
        "trace_path": ".agenticevals/agent-trace.jsonl",
    }


def _agent_prompt(task: TaskSpec, workspace: Path, computer) -> str:
    lines = [
        task.prompt,
        "",
        "You are being evaluated by agenticevals.",
        f"Workspace: {workspace}",
        "Write a machine-readable result to .agenticevals/result.json when finished:",
        '{"ok": true, "final_message": "short summary"}',
        "Optionally append JSONL events to .agenticevals/agent-trace.jsonl for commands, edits, tool calls, and observations.",
    ]
    sandbox_url = getattr(computer, "sandbox_url", None)
    if sandbox_url:
        lines.extend(
            [
                "",
                f"Sandbox HTTP computer interface: {sandbox_url}",
                "Available endpoints: POST /exec, /read, /write, /edit, /glob, /grep, /download, /browser/goto, /browser/check.",
                "Use the sandbox interface for observable computer actions when possible.",
            ]
        )
    if task.tools:
        lines.append("")
        lines.append("Declared task tools are available in .agenticevals/task.json.")
    return "\n".join(lines)


def _load_declared_result(path: Path, trace: Trajectory) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        trace.add("agent.result_file.error", path=str(path), error=str(exc))
        return None
    if not isinstance(data, dict):
        trace.add("agent.result_file.error", path=str(path), error="result file must be a JSON object")
        return None
    return data
