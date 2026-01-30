from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from agenticevals.computer.context import ComputerContext
from agenticevals.environments import Environment
from agenticevals.rewards import Reward, RewardComponent
from agenticevals.rollouts import AgentResult


class AgentSmokeEnv(Environment):
    name = "agent-smoke"

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self._items: list[dict[str, Any]] = []
        self._index = 0

    def setup(self) -> None:
        self.settings.ensure_dirs()
        root = self.settings.workspace_path / "agent-smoke-fixtures"
        root.mkdir(parents=True, exist_ok=True)
        self._items = [_build_item(root, i) for i in range(100)]
        self._index = 0

    def get_next_item(self) -> dict[str, Any] | None:
        if self._index >= len(self._items):
            return None
        item = self._items[self._index]
        self._index += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        return str(item["prompt"])

    def fixture_path(self, item: dict[str, Any]) -> Path:
        return Path(str(item["fixture_path"]))

    def compute_reward(self, item: dict[str, Any], result: AgentResult, ctx: ComputerContext) -> Reward:
        kind = str(item["kind"])
        if kind == "browser":
            return _browser_reward(item, ctx)
        checks = [
            _file_equals(ctx, str(item["target_path"]), str(item["expected"])),
            _command_passes(ctx, str(item.get("check_command", "python3 -m unittest discover -s tests"))),
            _changed_required(ctx, str(item["target_path"])),
        ]
        passed = all(component.passed for component in checks)
        return Reward(
            value=sum(component.value for component in checks),
            max_value=sum(component.max_value for component in checks),
            passed=passed,
            components=checks,
        )


def _build_item(root: Path, index: int) -> dict[str, Any]:
    group = index // 20
    slot = index % 20
    item_id = f"{group}-{slot:02d}"
    fixture = root / item_id
    fixture.mkdir(parents=True, exist_ok=True)
    if group == 0:
        return _file_transform_item(fixture, item_id, slot)
    if group == 1:
        return _shell_reasoning_item(fixture, item_id, slot)
    if group == 2:
        return _debugging_item(fixture, item_id, slot)
    if group == 3:
        return _data_cleaning_item(fixture, item_id, slot)
    return _browser_item(fixture, item_id, slot)


def _file_transform_item(fixture: Path, item_id: str, slot: int) -> dict[str, Any]:
    source = f"alpha-{slot}\nbeta-{slot}\n"
    expected = source.upper()
    (fixture / "notes.txt").write_text(source, encoding="utf-8")
    return {
        "id": f"file-transform-{item_id}",
        "kind": "file",
        "fixture_path": str(fixture),
        "target_path": "notes.txt",
        "expected": expected,
        "prompt": "Uppercase every line in notes.txt.",
        "script": [{"action": "write_file", "path": "notes.txt", "content": expected}],
        "check_command": "test -f notes.txt",
    }


def _shell_reasoning_item(fixture: Path, item_id: str, slot: int) -> dict[str, Any]:
    values = [slot + 1, slot + 3, slot + 5]
    expected = str(sum(values)) + "\n"
    (fixture / "numbers.txt").write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")
    return {
        "id": f"shell-reasoning-{item_id}",
        "kind": "shell",
        "fixture_path": str(fixture),
        "target_path": "sum.txt",
        "expected": expected,
        "prompt": "Read numbers.txt and write their sum to sum.txt.",
        "script": [{"action": "write_file", "path": "sum.txt", "content": expected}],
        "check_command": "test -s sum.txt",
    }


def _debugging_item(fixture: Path, item_id: str, slot: int) -> dict[str, Any]:
    (fixture / "src").mkdir(exist_ok=True)
    (fixture / "tests").mkdir(exist_ok=True)
    (fixture / "src" / "mathlib.py").write_text(
        "def scale(value):\n    return value * 1\n",
        encoding="utf-8",
    )
    (fixture / "tests" / "test_mathlib.py").write_text(
        f"import unittest\nfrom src.mathlib import scale\n\nclass MathLibTests(unittest.TestCase):\n    def test_scale(self):\n        self.assertEqual(scale({slot + 2}), {(slot + 2) * 2})\n",
        encoding="utf-8",
    )
    expected = "def scale(value):\n    return value * 2\n"
    return {
        "id": f"debugging-{item_id}",
        "kind": "debugging",
        "fixture_path": str(fixture),
        "target_path": "src/mathlib.py",
        "expected": expected,
        "prompt": "Fix the Python bug so the unit test passes.",
        "script": [{"action": "write_file", "path": "src/mathlib.py", "content": expected}],
        "check_command": "python3 -m unittest discover -s tests",
    }


def _data_cleaning_item(fixture: Path, item_id: str, slot: int) -> dict[str, Any]:
    rows = [{"name": "Ada", "score": str(10 + slot)}, {"name": "", "score": "bad"}, {"name": "Lin", "score": str(20 + slot)}]
    with (fixture / "raw.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "score"])
        writer.writeheader()
        writer.writerows(rows)
    expected = f"name,score\nAda,{10 + slot}\nLin,{20 + slot}\n"
    return {
        "id": f"data-cleaning-{item_id}",
        "kind": "data",
        "fixture_path": str(fixture),
        "target_path": "clean.csv",
        "expected": expected,
        "prompt": "Create clean.csv by keeping rows from raw.csv with non-empty names and numeric scores.",
        "script": [{"action": "write_file", "path": "clean.csv", "content": expected}],
        "check_command": "test -s clean.csv",
    }


def _browser_item(fixture: Path, item_id: str, slot: int) -> dict[str, Any]:
    (fixture / "index.html").write_text(
        f"<!doctype html><title>Task {slot}</title><h1>Waiting</h1><p>Patch me.</p>",
        encoding="utf-8",
    )
    expected = f"<!doctype html><title>Task {slot}</title><h1>Ready {slot}</h1><p>Browser visible.</p>"
    return {
        "id": f"browser-{item_id}",
        "kind": "browser",
        "fixture_path": str(fixture),
        "target_path": "index.html",
        "expected": expected,
        "prompt": "Patch index.html so the browser-visible page says it is ready.",
        "script": [{"action": "write_file", "path": "index.html", "content": expected}],
    }


def _file_equals(ctx: ComputerContext, path: str, expected: str) -> RewardComponent:
    actual = ctx.read_file(path)
    passed = actual == expected
    return RewardComponent("file_equals", 1.0 if passed else 0.0, 1.0, passed, path)


def _command_passes(ctx: ComputerContext, command: str) -> RewardComponent:
    result = ctx.terminal(command, timeout=30)
    return RewardComponent("command_passes", 1.0 if result.ok else 0.0, 1.0, result.ok, f"returncode={result.returncode}")


def _changed_required(ctx: ComputerContext, path: str) -> RewardComponent:
    changed = ctx.changed_files()
    passed = path in changed
    return RewardComponent("changed_required", 1.0 if passed else 0.0, 1.0, passed, f"changed={changed}")


def _browser_reward(item: dict[str, Any], ctx: ComputerContext) -> Reward:
    browser = ctx.browser(base_url=ctx.workspace.as_uri() + "/", timeout=5)
    snapshot = browser.goto("index.html")
    artifact = browser.save_snapshot("agent-smoke-browser")
    expected = str(item["expected"])
    file_component = _file_equals(ctx, str(item["target_path"]), expected)
    visible = "Browser visible." in snapshot.text and "Ready" in snapshot.text
    browser_component = RewardComponent("browser_visible", 1.0 if visible else 0.0, 1.0, visible, f"snapshot={artifact}")
    changed_component = _changed_required(ctx, str(item["target_path"]))
    components = [file_component, browser_component, changed_component]
    return Reward(
        value=sum(component.value for component in components),
        max_value=sum(component.max_value for component in components),
        passed=all(component.passed for component in components),
        components=components,
    )
