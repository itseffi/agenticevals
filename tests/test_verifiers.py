import json
import tempfile
import unittest
from pathlib import Path

from agenticevals.config import Settings
from agenticevals.runner import run_task
from agenticevals.schema import AgentSpec, TaskSpec, VerifierSpec, WorkspaceSpec
from agenticevals.trace import Trajectory
from agenticevals.trajectory_export import build_typed_trajectory
from agenticevals.verifiers import REWARD_DETAILS_SCHEMA_VERSION, REWARD_SCHEMA_VERSION, VerifierContext, run_verifiers


def _context(task: TaskSpec, trace: Trajectory) -> VerifierContext:
    return VerifierContext(
        task=task,
        workspace=Path("."),
        trajectory=build_typed_trajectory(trace, task=task),
        raw_trace=trace,
        changed_files=[],
        command_results=[],
        file_results=[],
        browser_results=[],
        audit_data={},
        dispatches=[],
        final_response="Done.",
    )


class VerifierTests(unittest.TestCase):
    def test_runner_writes_reward_artifacts_from_default_verifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture"
            fixture.mkdir()
            task = TaskSpec.from_dict(
                {
                    "id": "reward-artifacts",
                    "title": "Reward artifacts",
                    "prompt": "Write answer.txt.",
                    "workspace": {"fixture_path": str(fixture)},
                    "agent": {"kind": "scripted", "script": [{"action": "write_file", "path": "answer.txt", "content": "done\n"}]},
                    "checks": {"files": [{"path": "answer.txt", "contains": "done"}]},
                }
            )

            result = run_task(task, Settings.from_env(root=root))
            reward = json.loads((result.run_dir / "reward.json").read_text(encoding="utf-8"))
            details = json.loads((result.run_dir / "reward-details.json").read_text(encoding="utf-8"))
            score_exists = (result.run_dir / "score.json").exists()

        self.assertTrue(result.score.passed)
        self.assertEqual(reward["schema_version"], REWARD_SCHEMA_VERSION)
        self.assertEqual(details["schema_version"], REWARD_DETAILS_SCHEMA_VERSION)
        self.assertTrue(any(component["name"] == "file:file:answer.txt" for component in reward["components"]))
        self.assertTrue(score_exists)

    def test_tool_call_verifier_uses_typed_trajectory(self):
        task = TaskSpec(
            id="tool-policy",
            title="Tool policy",
            prompt="Do not send email.",
            workspace=WorkspaceSpec(fixture_path="."),
            agent=AgentSpec(kind="noop"),
            verifiers=[VerifierSpec(type="tool_calls", config={"forbidden_tools": ["send_email"]})],
        )
        trace = Trajectory(task_id=task.id)
        trace.add("agent.tool_call.parsed", step=1, tool_name="send_email", tool_use_id="tu_1", arguments={"to": "a@example.com"})

        result = run_verifiers(_context(task, trace))

        self.assertFalse(result.passed)
        self.assertEqual(result.criteria[0].name, "tool_calls:forbidden:send_email")
        self.assertIn("count=1", result.criteria[0].detail)

    def test_trajectory_and_llm_rubric_verifiers_can_be_explicit(self):
        task = TaskSpec(
            id="explicit-verifiers",
            title="Explicit verifiers",
            prompt="Return done.",
            workspace=WorkspaceSpec(fixture_path="."),
            agent=AgentSpec(kind="noop"),
            verifiers=[
                VerifierSpec(type="trajectory_check", config={"require_final_message": True}),
                VerifierSpec(type="llm_rubric", name="rubric", config={"fixture_score": 0.8, "threshold": 0.7}),
            ],
        )
        trace = Trajectory(task_id=task.id)
        trace.add("agent.result", ok=True, final_message="Done.", metadata={})

        result = run_verifiers(_context(task, trace))

        self.assertTrue(result.passed)
        self.assertEqual({criterion.verifier_type for criterion in result.criteria}, {"trajectory_check", "llm_rubric"})
        self.assertAlmostEqual(result.reward, 0.9)


if __name__ == "__main__":
    unittest.main()
