import tempfile
import unittest
from pathlib import Path

from agenticevals.config import Settings
from examples.agent_smoke_env import AgentSmokeEnv


class DeterminismTests(unittest.TestCase):
    def test_agent_smoke_generates_same_items_for_same_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings.from_env(root=Path(tmp))
            first = AgentSmokeEnv(settings)
            first.setup()
            first_ids = [first.get_next_item()["id"] for _ in range(10)]

            second = AgentSmokeEnv(settings)
            second.setup()
            second_ids = [second.get_next_item()["id"] for _ in range(10)]

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_ids[0], "file-transform-0-00")


if __name__ == "__main__":
    unittest.main()
