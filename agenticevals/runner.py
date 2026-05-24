from __future__ import annotations

import json
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

from .agents.factory import create_agent
from .computer.browser import run_browser_checks
from .computer.files import run_file_checks
from .computer.shell import Shell
from .config import Settings
from .reporting import write_html_report, write_json_report
from .sandbox import SandboxClient, SandboxServer
from .schema import TaskSpec
from .scorers import EvalScore, ScoreItem
from .scoring import DimensionScores, score_dimensions
from .services import ServiceManager
from .tools import ToolDispatcher, ToolResult
from .trace import Trajectory
from .trajectory_export import build_typed_trajectory, write_typed_trajectory
from .utils import CommandResult, safe_relative_path
from .verifiers import VerifierContext, run_verifiers, score_to_verifier_result, write_reward_artifacts
from .workspace import WorkspaceManager


@dataclass(frozen=True)
class RunResult:
    task: TaskSpec
    run_dir: Path
    workspace: Path
    trace_path: Path
    report_path: Path
    score: EvalScore
    changed_files: list[str]
    dimensions: DimensionScores | None = None


class AgentRuntime:
    def __init__(self, shell: Shell, dispatcher: ToolDispatcher | None = None, sandbox: SandboxClient | None = None):
        self.shell = shell
        self.dispatcher = dispatcher
        self.sandbox = sandbox
        self.sandbox_url = sandbox.base_url if sandbox is not None else None

    def terminal(self, command: str, timeout: int | None = None):
        if self.sandbox is not None:
            result = self.sandbox.exec(command, timeout_seconds=timeout or 60)
            command_result = CommandResult(
                command=command,
                returncode=int(result.get("exit_code", 1)),
                stdout=str(result.get("stdout", "")),
                stderr=str(result.get("stderr", "")),
                timed_out=result.get("exit_code") == -1,
            )
            self.shell.trace.add(
                "agent.sandbox.exec",
                command=command,
                returncode=command_result.returncode,
                stdout=command_result.stdout[-4000:],
                stderr=command_result.stderr[-4000:],
                timed_out=command_result.timed_out,
            )
            return command_result
        return self.shell.run(command, timeout=timeout or 60, event_type="agent.shell")

    def read_file(self, path: str) -> str:
        rel = safe_relative_path(path)
        if self.sandbox is not None:
            payload = self.sandbox.read(str(rel))
            content = str(payload.get("content", ""))
            self.shell.trace.add("agent.sandbox.read", path=str(rel), bytes=len(content.encode("utf-8")))
            return content
        return (self.shell.cwd / rel).read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> None:
        rel = safe_relative_path(path)
        if self.sandbox is not None:
            self.sandbox.write(str(rel), content)
            self.shell.trace.add("agent.sandbox.write", path=str(rel), bytes=len(content.encode("utf-8")))
            return
        target = self.shell.cwd / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def dispatch_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        if self.dispatcher is None:
            raise RuntimeError("no tool dispatcher configured")
        return self.dispatcher.dispatch(tool_name, arguments)


def cap_task_steps(task: TaskSpec, settings: Settings) -> TaskSpec:
    """Clamp a task's step budget to the operator-wide AGENTICEVALS_AGENT_MAX_STEPS.

    Agent loops read ``task.limits.max_steps`` directly, so applying the cap here
    gives the setting effect across every adapter without threading it through each.
    """
    if task.limits.max_steps <= settings.agent_max_steps:
        return task
    return replace(task, limits=replace(task.limits, max_steps=settings.agent_max_steps))


def run_task(task: TaskSpec, settings: Settings, agent_override: str | None = None, *, use_sandbox_server: bool = False) -> RunResult:
    settings.ensure_dirs()
    task = cap_task_steps(task, settings)
    workspace = WorkspaceManager(settings.runs_path).create(task)
    _write_task_artifacts(task, workspace.run_dir)
    trace = Trajectory(task_id=task.id)
    trace.add("run.start", task_id=task.id, title=task.title, workspace=str(workspace.path))

    shell = Shell(workspace.path, trace)
    dispatcher: ToolDispatcher | None = None
    audit_data: dict = {}
    sandbox_context = SandboxServer(workspace.path, trace) if use_sandbox_server else nullcontext(None)
    with ServiceManager(task.services, cwd=_task_resource_root(task, settings.root), trace=trace) as services, sandbox_context as sandbox:
        services.reset_all()
        for command in task.workspace.setup:
            result = shell.run(command, timeout=settings.action_long_timeout, event_type="workspace.setup")
            if not result.ok:
                trace.add("run.abort", reason="setup_failed", command=command, returncode=result.returncode)
                return _finish_aborted_run(task, workspace, trace, f"setup failed: {command}", result.stderr or result.stdout)

        dispatcher = ToolDispatcher(task.tools, task.tool_endpoints, trace=trace, timeout=settings.action_short_timeout) if task.tools else None
        runtime = AgentRuntime(shell, dispatcher, sandbox.client if sandbox is not None else None)
        agent = create_agent(task, settings, override=agent_override)
        agent_timeout = max(1, task.limits.max_minutes) * 60
        agent_run = agent.run(task, workspace.path, trace, timeout=agent_timeout, computer=runtime)
        trace.add("agent.result", ok=agent_run.ok, final_message=agent_run.final_message, metadata=agent_run.metadata)
        _inject_grader_files(task, workspace.path, trace)
        _copy_local_grader_files(task, workspace.run_dir, trace)
        snapshots = _collect_env_snapshots(task, shell, workspace.run_dir, trace)
        audit_data = services.audit_all()

    command_results: list[tuple[str, bool, str]] = []
    for command in task.checks.commands:
        result = shell.run(command, timeout=settings.action_long_timeout, event_type="verify.command")
        detail = f"returncode={result.returncode}"
        if result.stderr:
            detail += f"; stderr={result.stderr[-500:]}"
        command_results.append((command, result.ok, detail))

    file_results = run_file_checks(workspace.path, task.checks.files)
    for item in file_results:
        trace.add("verify.file", name=item.name, passed=item.passed, detail=item.detail)

    dev_url = task.workspace.dev_server.get("url")
    browser_results = []
    dev_proc: subprocess.Popen | None = None
    try:
        dev_command = task.workspace.dev_server.get("command")
        if dev_command and task.checks.browser:
            trace.add("workspace.dev_server.start", command=dev_command, url=dev_url)
            dev_proc = subprocess.Popen(
                dev_command,
                cwd=str(workspace.path),
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(float(task.workspace.dev_server.get("startup_wait_seconds", 2)))
            trace.add("workspace.dev_server.started", pid=dev_proc.pid, returncode=dev_proc.poll())
        browser_results = run_browser_checks(
            dev_url,
            task.checks.browser,
            timeout=settings.action_short_timeout,
            artifact_dir=workspace.run_dir / "artifacts" / "browser",
        )
        for item in browser_results:
            trace.add("verify.browser", name=item.name, passed=item.passed, detail=item.detail)
    finally:
        if dev_proc is not None:
            trace.add("workspace.dev_server.stop", pid=dev_proc.pid)
            dev_proc.terminate()
            try:
                stdout, stderr = dev_proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                dev_proc.kill()
                stdout, stderr = dev_proc.communicate(timeout=5)
            trace.add(
                "workspace.dev_server.output",
                returncode=dev_proc.returncode,
                stdout=(stdout or "")[-4000:],
                stderr=(stderr or "")[-4000:],
            )

    changed_files = _filter_non_agent_changes(task, WorkspaceManager.changed_files(workspace.path))
    diff = WorkspaceManager.diff(workspace.path)
    (workspace.run_dir / "diff.patch").write_text(diff.stdout, encoding="utf-8")
    trace.add("git.changed_files", files=changed_files)

    dimensions = None
    if task.expected_actions or task.safety_checks or task.tools:
        dimensions = score_dimensions(
            audit_data=audit_data,
            dispatches=dispatcher.records if dispatcher is not None else [],
            expected_actions=task.expected_actions,
            safety_checks=task.safety_checks,
            final_response=agent_run.final_message,
        )
        (workspace.run_dir / "dimensions.json").write_text(json.dumps(dimensions.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        (workspace.run_dir / "audit.json").write_text(json.dumps(audit_data, indent=2, sort_keys=True), encoding="utf-8")
        trace.add("score.dimensions", **dimensions.to_dict())
    typed_for_verifiers = build_typed_trajectory(trace, task=task)
    verifier_result = run_verifiers(
        VerifierContext(
            task=task,
            workspace=workspace.path,
            trajectory=typed_for_verifiers,
            raw_trace=trace,
            changed_files=changed_files,
            command_results=command_results,
            file_results=file_results,
            browser_results=browser_results,
            audit_data=audit_data,
            dispatches=dispatcher.records if dispatcher is not None else [],
            final_response=agent_run.final_message,
        )
    )
    score = verifier_result.to_score()
    trace.add("score", **score.to_dict())
    trace.add("run.finish", passed=score.passed, points=score.points, max_points=score.max_points)

    trace_path = workspace.run_dir / "trajectory.jsonl"
    trace.write_jsonl(trace_path)
    write_typed_trajectory(trace, workspace.run_dir / "trajectory.json", task=task)
    write_reward_artifacts(workspace.run_dir, verifier_result)
    (workspace.run_dir / "score.json").write_text(json.dumps(score.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    report_path = workspace.run_dir / "report.html"
    write_html_report(report_path, task, score, trace, changed_files)
    write_json_report(workspace.run_dir / "report.json", task, score, trace, changed_files)

    return RunResult(
        task=task,
        run_dir=workspace.run_dir,
        workspace=workspace.path,
        trace_path=trace_path,
        report_path=report_path,
        score=score,
        changed_files=changed_files,
        dimensions=dimensions,
    )


def _score_with_dimensions(score: EvalScore, dimensions: DimensionScores) -> EvalScore:
    item = ScoreItem(
        name="dimensions:task_score",
        passed=dimensions.passed,
        points=dimensions.task_score * 100,
        max_points=100,
        detail=json.dumps(dimensions.to_dict(), sort_keys=True),
    )
    return EvalScore(
        passed=score.passed and dimensions.passed,
        points=score.points + item.points,
        max_points=score.max_points + item.max_points,
        items=score.items + [item],
    )


def _write_task_artifacts(task: TaskSpec, run_dir: Path) -> None:
    payload = {
        "schema_version": "agenticevals.task-artifact.v1",
        "task": task.to_dict(),
        "source_path": str(task.source_path) if task.source_path else None,
    }
    (run_dir / "task.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _filter_non_agent_changes(task: TaskSpec, changed_files: list[str]) -> list[str]:
    excluded: list[str] = []
    for rel in task.sandbox_grader_files:
        normalized = rel.strip("/")
        excluded.append(normalized)
    return [
        path
        for path in changed_files
        if not any(path == excluded_path or path.startswith(excluded_path.rstrip("/") + "/") for excluded_path in excluded)
    ]


def _finish_aborted_run(task: TaskSpec, workspace, trace: Trajectory, reason: str, detail: str) -> RunResult:
    changed_files = _filter_non_agent_changes(task, WorkspaceManager.changed_files(workspace.path))
    diff = WorkspaceManager.diff(workspace.path)
    (workspace.run_dir / "diff.patch").write_text(diff.stdout, encoding="utf-8")
    score = EvalScore(
        passed=False,
        points=0,
        max_points=100,
        items=[ScoreItem(name="setup", passed=False, points=0, max_points=100, detail=f"{reason}; {detail[-500:]}")],
    )
    trace.add("git.changed_files", files=changed_files)
    trace.add("score", **score.to_dict())
    trace.add("run.finish", passed=False, points=0, max_points=100, reason=reason)
    trace_path = workspace.run_dir / "trajectory.jsonl"
    trace.write_jsonl(trace_path)
    write_typed_trajectory(trace, workspace.run_dir / "trajectory.json", task=task)
    write_reward_artifacts(workspace.run_dir, score_to_verifier_result(score))
    (workspace.run_dir / "score.json").write_text(json.dumps(score.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    report_path = workspace.run_dir / "report.html"
    write_html_report(report_path, task, score, trace, changed_files)
    write_json_report(workspace.run_dir / "report.json", task, score, trace, changed_files)
    return RunResult(
        task=task,
        run_dir=workspace.run_dir,
        workspace=workspace.path,
        trace_path=trace_path,
        report_path=report_path,
        score=score,
        changed_files=changed_files,
    )


def _task_resource_root(task: TaskSpec, fallback: Path) -> Path:
    if task.source_path and task.source_path.parent.name == "tasks" and task.source_path.parent.parent.name == "configs":
        return task.source_path.parent.parent.parent
    return fallback


def _inject_grader_files(task: TaskSpec, workspace_path: Path, trace: Trajectory) -> None:
    if not task.sandbox_grader_files:
        return
    task_dir = task.source_path.parent if task.source_path else Path.cwd()
    for rel in task.sandbox_grader_files:
        source = (task_dir / rel).resolve()
        target = workspace_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        trace.add("grader_file.inject", source=str(source), target=str(target.relative_to(workspace_path)))


def _copy_local_grader_files(task: TaskSpec, run_dir: Path, trace: Trajectory) -> None:
    if not task.local_grader_files:
        return
    task_dir = task.source_path.parent if task.source_path else Path.cwd()
    target_dir = run_dir / "local_grader_files"
    target_dir.mkdir(parents=True, exist_ok=True)
    for rel in task.local_grader_files:
        source = (task_dir / rel).resolve()
        target = target_dir / Path(rel).name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        trace.add("grader_file.local_copy", source=str(source), target=str(target))


def _collect_env_snapshots(task: TaskSpec, shell: Shell, run_dir: Path, trace: Trajectory) -> dict:
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot: dict[str, object] = {"commands": [], "files": []}
    for command in task.env_snapshot_commands:
        result = shell.run(command, timeout=60, event_type="env_snapshot.command")
        snapshot["commands"].append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
                "timed_out": result.timed_out,
            }
        )
    for pattern in task.env_snapshot_files:
        matches = sorted(shell.cwd.glob(pattern) if not Path(pattern).is_absolute() else Path("/").glob(str(Path(pattern).relative_to("/"))))
        for source in matches:
            if not source.is_file():
                continue
            target = snapshot_dir / source.name
            shutil.copy2(source, target)
            snapshot["files"].append({"source": str(source), "saved_as": str(target)})
            trace.add("env_snapshot.file", source=str(source), saved_as=str(target))
    if task.env_snapshot_commands or task.env_snapshot_files:
        (snapshot_dir / "snapshot_index.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        trace.add("env_snapshot.finish", snapshot_dir=str(snapshot_dir))
    return snapshot
