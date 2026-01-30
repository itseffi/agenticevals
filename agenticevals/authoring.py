from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticevals.config import Settings
from agenticevals.runner import run_task
from agenticevals.schema import TaskSpec


@dataclass(frozen=True)
class QualityIssue:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def scaffold_task(settings: Settings, *, task_id: str, kind: str = "hidden-grader", force: bool = False) -> Path:
    safe_id = _safe_id(task_id)
    if kind != "hidden-grader":
        raise ValueError("supported scaffold kinds: hidden-grader")
    fixture_dir = settings.root / "examples" / safe_id
    grader_dir = settings.task_config_dir / "graders" / safe_id
    task_path = settings.task_config_dir / f"{safe_id}.json"
    for path in [fixture_dir, grader_dir, task_path]:
        if path.exists() and not force:
            raise FileExistsError(path)
    (fixture_dir / "src").mkdir(parents=True, exist_ok=True)
    grader_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "README.md").write_text(f"Fix `src/solution.py` for task `{safe_id}`.\n", encoding="utf-8")
    (fixture_dir / "src" / "__init__.py").write_text("", encoding="utf-8")
    (fixture_dir / "src" / "solution.py").write_text(
        "def transform(text: str) -> str:\n    return text\n",
        encoding="utf-8",
    )
    (grader_dir / "test_hidden.py").write_text(
        "import unittest\n\n"
        "from src.solution import transform\n\n\n"
        "class HiddenTests(unittest.TestCase):\n"
        "    def test_transform(self):\n"
        "        self.assertEqual(transform('agentic evals'), 'AGENTIC EVALS')\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    payload = {
        "id": safe_id,
        "title": f"Hidden grader scaffold for {safe_id}",
        "prompt": "Fix src/solution.py so transform returns the uppercase version of the input string.",
        "workspace": {"fixture_path": f"../../examples/{safe_id}"},
        "agent": {
            "kind": "scripted",
            "script": [
                {
                    "action": "replace",
                    "path": "src/solution.py",
                    "old": "def transform(text: str) -> str:\n    return text\n",
                    "new": "def transform(text: str) -> str:\n    return text.upper()\n",
                },
                {"action": "final", "message": "Updated transform to return uppercase text."},
            ],
        },
        "sandbox_grader_files": [f"graders/{safe_id}"],
        "checks": {"commands": [f"python -m unittest discover -s graders/{safe_id}"]},
        "policies": {"require_changed_files": ["src/solution.py"], "max_changed_files": 1},
        "limits": {"max_steps": 5, "max_minutes": 5},
    }
    task_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return task_path


def validate_task_quality(task_path: Path, settings: Settings, *, run: bool = True) -> tuple[bool, list[QualityIssue]]:
    task = TaskSpec.from_file(task_path.expanduser().resolve())
    issues: list[QualityIssue] = []
    fixture = task.resolve_fixture()
    issues.append(QualityIssue("fixture_exists", fixture.exists(), str(fixture)))
    issues.append(QualityIssue("has_checks", bool(task.checks.commands or task.checks.files or task.checks.browser), "task has at least one verifier"))
    issues.append(QualityIssue("has_policy", bool(task.policies.require_changed_files or task.policies.forbidden_paths or task.policies.max_changed_files is not None), "task has git policy"))
    issues.append(QualityIssue("hidden_grader_configured", bool(task.sandbox_grader_files), str(task.sandbox_grader_files)))
    for rel in task.sandbox_grader_files:
        visible = (fixture / rel).exists()
        source = (task.source_path.parent / rel).exists() if task.source_path else False
        issues.append(QualityIssue(f"hidden_grader_not_in_fixture:{rel}", not visible, "not visible before run" if not visible else "visible in fixture"))
        issues.append(QualityIssue(f"hidden_grader_source_exists:{rel}", source, str(task.source_path.parent / rel if task.source_path else rel)))
    if run:
        try:
            scripted = run_task(task, settings, agent_override="scripted")
            issues.append(QualityIssue("scripted_passes", scripted.score.passed, str(scripted.run_dir)))
        except Exception as exc:
            issues.append(QualityIssue("scripted_passes", False, str(exc)))
        try:
            noop = run_task(task, settings, agent_override="noop")
            issues.append(QualityIssue("noop_fails", not noop.score.passed, str(noop.run_dir)))
        except Exception as exc:
            issues.append(QualityIssue("noop_fails", False, str(exc)))
    return all(issue.passed for issue in issues), issues


def _safe_id(task_id: str) -> str:
    safe = re.sub(r"[^a-z0-9_-]+", "-", task_id.strip().lower()).strip("-")
    if not safe:
        raise ValueError("task id must contain at least one alphanumeric character")
    return safe
