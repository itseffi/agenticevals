from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from agenticevals.computer.browser import run_browser_checks
from agenticevals.computer.files import run_file_checks
from agenticevals.computer.shell import Shell
from agenticevals.runner import _filter_non_agent_changes
from agenticevals.schema import TaskSpec
from agenticevals.tools.types import ToolDispatchRecord
from agenticevals.trace import TraceEvent, Trajectory
from agenticevals.trajectory_export import build_typed_trajectory
from agenticevals.verifiers import VerifierContext, run_verifiers
from agenticevals.workspace import WorkspaceManager


def recompute_rewards(run_dir: Path) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if (root / "task.json").exists() and (root / "workspace").exists():
        result = _recompute_task_run(root)
    elif (root / "rollout.json").exists():
        rollout = _read_json(root / "rollout.json", {})
        result = {
            "schema_version": "agenticevals.reward-recompute.v1",
            "method": "stored_environment_reward",
            "recomputed": False,
            "reason": "environment rewards require the original environment implementation and item state",
            "reward": rollout.get("reward", {}),
        }
    elif (root / "score.json").exists():
        score = _read_json(root / "score.json", {})
        result = {
            "schema_version": "agenticevals.reward-recompute.v1",
            "method": "legacy_stored_score",
            "recomputed": False,
            "reason": "run does not contain task.json; new runs store task specs for recomputation",
            "score": score,
        }
    else:
        raise FileNotFoundError(f"no recomputable reward artifacts under {root}")
    (root / "recomputed_rewards.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _recompute_task_run(root: Path) -> dict[str, Any]:
    artifact = _read_json(root / "task.json", {})
    task = TaskSpec.from_dict(artifact["task"])
    workspace = root / "workspace"
    trace = Trajectory(task_id=task.id)
    shell = Shell(workspace, trace)
    command_results = []
    for command in task.checks.commands:
        result = shell.run(command, timeout=300, event_type="recompute.command")
        detail = f"returncode={result.returncode}"
        if result.stderr:
            detail += f"; stderr={result.stderr[-500:]}"
        command_results.append((command, result.ok, detail))
    file_results = run_file_checks(workspace, task.checks.files)
    browser_results = _recompute_browser(task, workspace, root)
    changed_files = _filter_non_agent_changes(task, WorkspaceManager.changed_files(workspace))
    raw_trace = _load_trace(root, task.id)
    typed = build_typed_trajectory(raw_trace, task=task)
    verifier_result = run_verifiers(
        VerifierContext(
            task=task,
            workspace=workspace,
            trajectory=typed,
            raw_trace=raw_trace,
            changed_files=changed_files,
            command_results=command_results,
            file_results=file_results,
            browser_results=browser_results,
            audit_data=_read_json(root / "audit.json", {}),
            dispatches=_dispatches_from_trace(root),
            final_response=_final_response_from_trace(root),
        )
    )
    score = verifier_result.to_score()
    reward = verifier_result.to_reward_dict()
    old_score = _read_json(root / "score.json", {})
    old_reward = _read_json(root / "reward.json", {})
    return {
        "schema_version": "agenticevals.reward-recompute.v1",
        "method": "task_workspace_recomputed",
        "recomputed": True,
        "task_id": task.id,
        "reward": reward,
        "previous_reward": old_reward,
        "score": score.to_dict(),
        "previous_score": old_score,
        "matches_previous": _reward_equivalent(reward, old_reward) if old_reward else _score_equivalent(score.to_dict(), old_score),
    }


def _recompute_browser(task: TaskSpec, workspace: Path, root: Path):
    if not task.checks.browser:
        return []
    dev_url = task.workspace.dev_server.get("url")
    dev_command = task.workspace.dev_server.get("command")
    proc: subprocess.Popen | None = None
    try:
        if dev_command:
            proc = subprocess.Popen(
                dev_command,
                cwd=str(workspace),
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(float(task.workspace.dev_server.get("startup_wait_seconds", 2)))
        return run_browser_checks(
            dev_url,
            task.checks.browser,
            timeout=60,
            artifact_dir=root / "artifacts" / "browser-recomputed",
        )
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def _dispatches_from_trace(root: Path) -> list[ToolDispatchRecord]:
    records: list[ToolDispatchRecord] = []
    for event in _trace_events(root):
        if event.get("type") != "tool.dispatch":
            continue
        data = event.get("data", {})
        records.append(
            ToolDispatchRecord(
                tool_name=str(data.get("tool_name", "")),
                request=dict(data.get("request", {})),
                status=int(data.get("status", 0) or 0),
                response=data.get("response", {}),
                latency_ms=float(data.get("latency_ms", 0.0) or 0.0),
                ok=bool(data.get("ok", False)),
                error=str(data.get("error", "")),
            )
        )
    return records


def _final_response_from_trace(root: Path) -> str:
    for event in _trace_events(root):
        if event.get("type") == "agent.result":
            return str(event.get("data", {}).get("final_message", ""))
    return ""


def _trace_events(root: Path) -> list[dict[str, Any]]:
    path = root / "trajectory.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_trace(root: Path, task_id: str) -> Trajectory:
    trace = Trajectory(task_id=task_id, run_id=root.name)
    trace.events = [
        TraceEvent(type=str(event.get("type", "unknown")), data=dict(event.get("data", {})), ts=float(event.get("ts", 0.0) or 0.0))
        for event in _trace_events(root)
    ]
    return trace


def _score_equivalent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        bool(a.get("passed")) == bool(b.get("passed"))
        and round(float(a.get("points", 0.0)), 6) == round(float(b.get("points", 0.0)), 6)
        and round(float(a.get("max_points", 0.0)), 6) == round(float(b.get("max_points", 0.0)), 6)
    )


def _reward_equivalent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        bool(a.get("passed")) == bool(b.get("passed"))
        and round(float(a.get("reward", 0.0)), 6) == round(float(b.get("reward", 0.0)), 6)
        and round(float(a.get("max_reward", 0.0)), 6) == round(float(b.get("max_reward", 0.0)), 6)
    )


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
