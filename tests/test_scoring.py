import unittest

from agenticevals.computer.files import FileCheckResult
from agenticevals.schema import AgentSpec, TaskSpec, WorkspaceSpec
from agenticevals.scorers import score_run


def _task() -> TaskSpec:
    return TaskSpec(
        id="scoring",
        title="Scoring",
        prompt="noop",
        workspace=WorkspaceSpec(fixture_path="."),
        agent=AgentSpec(kind="noop"),
    )


class ScoringTests(unittest.TestCase):
    def test_all_checks_passing_yields_passed_even_with_fractional_points(self):
        # Regression: counts whose even split drifts (1 command + 3 file checks
        # sum to 100.00000000000001). points and max_points drift identically
        # when everything passes, so `points == max_points` stays correct.
        command_results = [("build", True, "ok")]
        file_results = [
            FileCheckResult(name=f"file:{i}", passed=True, detail="ok") for i in range(3)
        ]
        score = score_run(_task(), command_results, file_results, [], [])
        self.assertTrue(all(item.passed for item in score.items))
        self.assertTrue(score.passed)

    def test_any_failing_check_yields_not_passed(self):
        file_results = [
            FileCheckResult(name="file:0", passed=True, detail="ok"),
            FileCheckResult(name="file:1", passed=False, detail="bad"),
        ]
        score = score_run(_task(), [], file_results, [], [])
        self.assertFalse(score.passed)


if __name__ == "__main__":
    unittest.main()
