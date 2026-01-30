from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticevals.computer.context import ComputerContext
from agenticevals.environments import Environment
from agenticevals.rewards import Reward, RewardComponent
from agenticevals.rollouts import AgentResult


class PatchPythonBugEnv(Environment):
    name = "patch-python-bug-env"

    def setup(self) -> None:
        self._items = [
            {
                "id": "divide-by-zero",
                "title": "Patch divide-by-zero behavior",
                "prompt": (
                    "Fix the divide-by-zero behavior in src/calculator.py. "
                    "The function should raise ZeroDivisionError when b is zero. "
                    "Do not delete or weaken tests. Run the tests before finishing."
                ),
                "fixture": Path(__file__).resolve().parents[1] / "patch-python-bug" / "fixtures",
                "script": [
                    {
                        "action": "replace",
                        "path": "src/calculator.py",
                        "old": "    if b == 0:\n        return 0\n    return a / b\n",
                        "new": "    if b == 0:\n        raise ZeroDivisionError(\"division by zero\")\n    return a / b\n",
                    },
                    {
                        "action": "run",
                        "command": "python3 -m unittest discover -s tests",
                        "must_pass": True,
                    },
                    {
                        "action": "final",
                        "message": "Fixed divide-by-zero behavior and verified the unittest suite passes.",
                    },
                ],
            }
        ]
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
        return Path(item["fixture"])

    def compute_reward(self, item: dict[str, Any], result: AgentResult, ctx: ComputerContext) -> Reward:
        test = ctx.terminal("python3 -m unittest discover -s tests", timeout=60)
        implementation = ctx.read_file("src/calculator.py")
        changed = ctx.changed_files()

        components = [
            RewardComponent(
                name="tests_pass",
                value=0.45 if test.ok else 0.0,
                max_value=0.45,
                passed=test.ok,
                detail=f"returncode={test.returncode}",
            ),
            RewardComponent(
                name="implementation_raises",
                value=0.25 if "raise ZeroDivisionError" in implementation else 0.0,
                max_value=0.25,
                passed="raise ZeroDivisionError" in implementation,
                detail="src/calculator.py contains raise ZeroDivisionError",
            ),
            RewardComponent(
                name="tests_untouched",
                value=0.15 if not any(path.startswith("tests/") or path == "tests" for path in changed) else 0.0,
                max_value=0.15,
                passed=not any(path.startswith("tests/") or path == "tests" for path in changed),
                detail=f"changed_files={changed}",
            ),
            RewardComponent(
                name="minimal_change",
                value=0.15 if changed == ["src/calculator.py"] else 0.0,
                max_value=0.15,
                passed=changed == ["src/calculator.py"],
                detail=f"changed_files={changed}",
            ),
        ]
        return Reward.from_components(components)

