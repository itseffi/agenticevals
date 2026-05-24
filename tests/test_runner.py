import dataclasses
import tempfile
import unittest
from pathlib import Path

from agenticevals.config import Settings
from agenticevals.runner import cap_task_steps, run_task
from agenticevals.schema import AgentSpec, LimitsSpec, TaskSpec, WorkspaceSpec


def _cap_task(max_steps: int) -> TaskSpec:
    return TaskSpec(
        id="cap",
        title="Cap",
        prompt="noop",
        workspace=WorkspaceSpec(fixture_path="."),
        agent=AgentSpec(kind="noop"),
        limits=LimitsSpec(max_steps=max_steps),
    )


class CapStepsTests(unittest.TestCase):
    def test_caps_task_steps_to_agent_max_steps(self):
        settings = dataclasses.replace(Settings.from_env(root=Path(".")), agent_max_steps=10)
        capped = cap_task_steps(_cap_task(100), settings)
        self.assertEqual(capped.limits.max_steps, 10)

    def test_leaves_task_below_cap_untouched(self):
        settings = dataclasses.replace(Settings.from_env(root=Path(".")), agent_max_steps=50)
        task = _cap_task(5)
        self.assertIs(cap_task_steps(task, settings), task)


class RunnerTests(unittest.TestCase):
    def test_setup_failure_aborts_before_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture"
            fixture.mkdir()
            (fixture / "README.md").write_text("fixture\n", encoding="utf-8")
            task = TaskSpec.from_dict(
                {
                    "id": "setup-fails",
                    "title": "Setup fails",
                    "prompt": "Create answer.txt.",
                    "workspace": {
                        "fixture_path": str(fixture),
                        "setup": ["python -c 'import sys; sys.exit(7)'"],
                    },
                    "agent": {
                        "kind": "scripted",
                        "script": [{"action": "write_file", "path": "answer.txt", "content": "done\n"}],
                    },
                }
            )
            settings = Settings.from_env(root=root)
            result = run_task(task, settings)

            self.assertFalse(result.score.passed)
            self.assertEqual(result.score.items[0].name, "setup")
            self.assertFalse((result.workspace / "answer.txt").exists())
            trajectory = result.trace_path.read_text(encoding="utf-8")
            self.assertIn('"type": "run.abort"', trajectory)
            self.assertNotIn('"type": "agent.start"', trajectory)


if __name__ == "__main__":
    unittest.main()
