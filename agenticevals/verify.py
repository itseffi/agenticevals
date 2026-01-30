from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from agenticevals.config import Settings
from agenticevals.runner import run_task
from agenticevals.schema import TaskSpec


def verify_install() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rows.append(_check("python", True, sys.version.split()[0]))
    try:
        version = importlib.metadata.version("agenticevals")
        rows.append(_check("package_metadata", True, version))
    except importlib.metadata.PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        rows.append(_check("package_metadata", pyproject.exists(), "source checkout" if pyproject.exists() else "package is importable from source but not installed"))
    proc = subprocess.run([sys.executable, "-m", "agenticevals", "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    rows.append(_check("module_cli", proc.returncode == 0, (proc.stdout or proc.stderr)[:200]))
    return {"passed": all(row["passed"] for row in rows), "checks": rows}


def verify_live_adapters(settings: Settings, *, task_path: Path | None = None) -> dict[str, Any]:
    task = TaskSpec.from_file(task_path or settings.task_config_dir / "model-loop-write-file.json")
    rows = []
    for adapter, executable in [("codex", "codex"), ("claude-code", "claude")]:
        if shutil.which(executable) is None:
            rows.append(_check(adapter, False, f"{executable} executable not found"))
            continue
        try:
            result = run_task(task, settings, agent_override=adapter)
            detail = str(result.run_dir) if result.score.passed else _failure_detail(result.run_dir)
            rows.append(_check(adapter, result.score.passed, detail))
        except Exception as exc:
            rows.append(_check(adapter, False, str(exc)))
    return {"passed": all(row["passed"] for row in rows), "checks": rows}


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _failure_detail(run_dir: Path) -> str:
    parts = [str(run_dir)]
    trace_path = run_dir / "trajectory.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "agent.result":
                data = event.get("data", {})
                if not data.get("ok", False):
                    message = str(data.get("final_message", "")).strip().replace("\n", " ")
                    if message:
                        parts.append(message[:240])
                break
    score_path = run_dir / "score.json"
    if score_path.exists():
        score = json.loads(score_path.read_text(encoding="utf-8"))
        failed = [str(item.get("name")) for item in score.get("items", []) if not item.get("passed", True)]
        if failed:
            parts.append("failed checks: " + ", ".join(failed))
    return " | ".join(parts)
