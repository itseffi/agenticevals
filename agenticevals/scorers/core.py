from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agenticevals.computer.browser import BrowserCheckResult
from agenticevals.computer.files import FileCheckResult
from agenticevals.schema import TaskSpec


@dataclass(frozen=True)
class ScoreItem:
    name: str
    passed: bool
    points: float
    max_points: float
    detail: str


@dataclass(frozen=True)
class EvalScore:
    passed: bool
    points: float
    max_points: float
    items: list[ScoreItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "points": self.points,
            "max_points": self.max_points,
            "items": [asdict(item) for item in self.items],
        }


def _even_items(category: str, max_points: int, checks: list[tuple[str, bool, str]]) -> list[ScoreItem]:
    if not checks:
        return [ScoreItem(name=category, passed=True, points=max_points, max_points=max_points, detail="no checks configured")]
    each = max_points / len(checks)
    return [
        ScoreItem(name=f"{category}:{name}", passed=passed, points=each if passed else 0, max_points=each, detail=detail)
        for name, passed, detail in checks
    ]


def score_git_policy(task: TaskSpec, changed_files: list[str]) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    for forbidden in task.policies.forbidden_paths:
        touched = [path for path in changed_files if path == forbidden or path.startswith(forbidden.rstrip("/") + "/")]
        checks.append((f"forbidden:{forbidden}", not touched, f"touched={touched}"))
    if task.policies.max_changed_files is not None:
        checks.append(
            (
                "max_changed_files",
                len(changed_files) <= task.policies.max_changed_files,
                f"changed={len(changed_files)}, limit={task.policies.max_changed_files}",
            )
        )
    for required in task.policies.require_changed_files:
        checks.append((f"required:{required}", required in changed_files, f"changed_files={changed_files}"))
    return checks


def score_run(
    task: TaskSpec,
    command_results: list[tuple[str, bool, str]],
    file_results: list[FileCheckResult],
    browser_results: list[BrowserCheckResult],
    changed_files: list[str],
) -> EvalScore:
    items: list[ScoreItem] = []
    items += _even_items("command", task.score.command_checks, command_results)
    items += _even_items("file", task.score.file_checks, [(r.name, r.passed, r.detail) for r in file_results])
    items += _even_items("browser", task.score.browser_checks, [(r.name, r.passed, r.detail) for r in browser_results])
    items += _even_items("git_policy", task.score.git_policy, score_git_policy(task, changed_files))
    points = sum(item.points for item in items)
    max_points = sum(item.max_points for item in items)
    return EvalScore(passed=points == max_points, points=points, max_points=max_points, items=items)

