import tempfile
import unittest
from pathlib import Path

from agenticevals.config import Settings
from agenticevals.runner import run_task
from agenticevals.schema import TaskSpec


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
