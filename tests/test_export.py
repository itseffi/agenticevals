import json
import tempfile
import unittest
from pathlib import Path

from agenticevals.export import _reward_rows


class RewardRowDimensionTests(unittest.TestCase):
    def _rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "r"
            run.mkdir()
            (run / "dimensions.json").write_text(
                json.dumps(
                    {"completion": 1.0, "robustness": 1.0, "communication": 0.5, "safety": 1.0, "details": {}}
                ),
                encoding="utf-8",
            )
            rollout = {"source_rollout_path": str(run / "rollout.json"), "reward": {}}
            return _reward_rows(root, [rollout])

    def test_partial_dimension_keeps_its_fractional_score(self):
        rows = self._rows()
        comm = next(r for r in rows if r["component"] == "communication")
        self.assertEqual(comm["value"], 0.5)
        self.assertEqual(comm["score"], 0.5)
        self.assertFalse(comm["passed"])
        self.assertTrue(comm["partial"])

    def test_fully_met_dimension_is_passed_not_partial(self):
        rows = self._rows()
        completion = next(r for r in rows if r["component"] == "completion")
        self.assertTrue(completion["passed"])
        self.assertFalse(completion["partial"])


if __name__ == "__main__":
    unittest.main()
