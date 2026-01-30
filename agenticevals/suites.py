from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticevals.config import Settings
from agenticevals.failures import aggregate_failures
from agenticevals.runner import run_task
from agenticevals.schema import TaskSpec
from agenticevals.stats import bootstrap_ci
from agenticevals.trials import run_trials


@dataclass(frozen=True)
class SuiteTask:
    path: Path
    agent: str | None = None
    trials: int = 1
    sandbox_server: bool = False


@dataclass(frozen=True)
class SuiteSpec:
    id: str
    title: str
    tasks: list[SuiteTask]


def load_suite(path: Path) -> SuiteSpec:
    root = path.expanduser().resolve()
    data = json.loads(root.read_text(encoding="utf-8"))
    tasks = []
    for item in data.get("tasks", []):
        task_path = Path(str(item["path"]))
        if not task_path.is_absolute():
            task_path = (root.parent / task_path).resolve()
        tasks.append(
            SuiteTask(
                path=task_path,
                agent=item.get("agent"),
                trials=int(item.get("trials", data.get("trials", 1))),
                sandbox_server=bool(item.get("sandbox_server", data.get("sandbox_server", False))),
            )
        )
    return SuiteSpec(id=str(data["id"]), title=str(data["title"]), tasks=tasks)


def run_suite(path: Path, settings: Settings, *, agent_override: str | None = None, workers: int = 1, resume: Path | None = None) -> dict[str, Any]:
    suite = load_suite(path)
    if resume:
        run_dir = resume.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = settings.runs_path / f"suite-{suite.id}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = run_dir / "checkpoint.jsonl"
    rows = _load_checkpoint(checkpoint)
    done = {row.get("suite_key") or row["task_path"] for row in rows}
    pending = [suite_task for suite_task in suite.tasks if _suite_task_key(suite_task, agent_override) not in done]
    if workers <= 1:
        for suite_task in pending:
            row = _run_suite_task(suite_task, settings, agent_override)
            rows.append(row)
            _append_checkpoint(checkpoint, row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_suite_task, suite_task, settings, agent_override) for suite_task in pending]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                _append_checkpoint(checkpoint, row)
    summary = {
        "id": suite.id,
        "title": suite.title,
        "run_dir": str(run_dir),
        "total": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "pass_rate": sum(1 for row in rows if row["passed"]) / len(rows) if rows else 0.0,
        "mean_score": sum(float(row["score"]) for row in rows) / len(rows) if rows else 0.0,
        "tasks": rows,
    }
    (run_dir / "suite.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(_result_rows(rows), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "failures.json").write_text(json.dumps(aggregate_failures(rows), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _run_suite_task(suite_task: SuiteTask, settings: Settings, agent_override: str | None) -> dict[str, Any]:
    task = TaskSpec.from_file(suite_task.path)
    agent = agent_override or suite_task.agent
    suite_key = _suite_task_key(suite_task, agent_override)
    if suite_task.trials > 1:
        result = run_trials(task, settings, agent_override=agent, trials=suite_task.trials, use_sandbox_server=suite_task.sandbox_server)
        summary = result.to_dict()
        return {
            "suite_key": suite_key,
            "task_id": task.id,
            "task_path": str(suite_task.path),
            "agent": agent or task.agent.kind,
            "trials": suite_task.trials,
            "passed": bool(summary["pass_power_k"]),
            "score": float(summary["mean_score"]),
            "run_dir": str(result.run_dir),
            "summary": summary,
        }
    result = run_task(task, settings, agent_override=agent, use_sandbox_server=suite_task.sandbox_server)
    return {
        "suite_key": suite_key,
        "task_id": task.id,
        "task_path": str(suite_task.path),
        "agent": agent or task.agent.kind,
        "trials": 1,
        "passed": result.score.passed,
        "score": result.score.points / result.score.max_points if result.score.max_points else 0.0,
        "run_dir": str(result.run_dir),
        "summary": result.score.to_dict(),
    }


def _suite_task_key(suite_task: SuiteTask, agent_override: str | None) -> str:
    agent = agent_override or suite_task.agent or ""
    return json.dumps(
        {
            "path": str(suite_task.path),
            "agent": agent,
            "trials": suite_task.trials,
            "sandbox_server": suite_task.sandbox_server,
        },
        sort_keys=True,
    )


def _load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_agent.setdefault(str(row["agent"]), []).append(row)
    result_rows = []
    for agent, agent_rows in by_agent.items():
        result_rows.append(
            {
                "agent": agent,
                "tasks": len(agent_rows),
                "passed": sum(1 for row in agent_rows if row["passed"]),
                "pass_rate": sum(1 for row in agent_rows if row["passed"]) / len(agent_rows),
                "pass_rate_ci": bootstrap_ci([bool(row["passed"]) for row in agent_rows], seed=sum(ord(char) for char in agent)),
                "mean_score": sum(float(row["score"]) for row in agent_rows) / len(agent_rows),
                "mean_score_ci": bootstrap_ci([float(row["score"]) for row in agent_rows], seed=sum(ord(char) for char in agent + ":score")),
            }
        )
    return sorted(result_rows, key=lambda row: (-row["pass_rate"], -row["mean_score"], row["agent"]))
