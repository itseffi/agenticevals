import json
import tempfile
import unittest
from pathlib import Path

from agenticevals.agents.claude_loop import ClaudeAgent
from agenticevals.config import Settings
from agenticevals.runner import run_task
from agenticevals.schema import AgentSpec, LimitsSpec, TaskSpec, WorkspaceSpec
from agenticevals.trace import Trajectory
from agenticevals.trajectory_export import build_typed_trajectory
from agenticevals.trajectory_schema import TRAJECTORY_SCHEMA_VERSION, trajectory_semantic_hash
from agenticevals.trajectory_validate import validate_typed_trajectory


class _FakeCommandResult:
    def __init__(self, ok=True, returncode=0, stdout="", stderr=""):
        self.ok = ok
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeComputer:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def terminal(self, command: str, timeout: int = 60) -> _FakeCommandResult:
        return _FakeCommandResult(ok=True, returncode=0, stdout="ok", stderr="")

    def read_file(self, path: str) -> str:
        return (self.workspace / path).read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        target = self.workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _scripted_provider(script):
    state = {"index": 0}

    def provider(messages, tools, system):
        index = state["index"]
        state["index"] = index + 1
        if index >= len(script):
            return {"content": [{"type": "text", "text": "Done."}], "stop_reason": "end_turn", "usage": {}}
        return script[index]

    return provider


def _claude_task() -> TaskSpec:
    return TaskSpec(
        id="typed-claude",
        title="Typed Claude",
        prompt="Read input.txt and report it.",
        workspace=WorkspaceSpec(fixture_path="."),
        agent=AgentSpec(kind="claude", model="fixture"),
        limits=LimitsSpec(max_steps=5, max_minutes=1),
    )


class TypedTrajectoryTests(unittest.TestCase):
    def test_claude_fixture_exports_valid_typed_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "input.txt").write_text("hello", encoding="utf-8")
            task = _claude_task()
            trace = Trajectory(task_id=task.id)
            script = [
                {
                    "content": [
                        {"type": "text", "text": "Reading."},
                        {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "input.txt"}},
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
                {
                    "content": [{"type": "text", "text": "hello"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 150, "output_tokens": 8},
                },
            ]
            ClaudeAgent(fixture_provider=_scripted_provider(script)).run(task, workspace, trace, timeout=30, computer=_FakeComputer(workspace))

        typed = build_typed_trajectory(trace, task=task)
        payload = typed.to_dict()
        validation = validate_typed_trajectory(payload)
        self.assertTrue(validation.ok, validation.errors)
        self.assertEqual(payload["schema_version"], TRAJECTORY_SCHEMA_VERSION)
        self.assertEqual(payload["final_metrics"]["total_input_tokens"], 250)
        self.assertEqual(payload["final_metrics"]["total_output_tokens"], 28)
        calls = [call for step in payload["steps"] for call in step.get("tool_calls", [])]
        results = [result for step in payload["steps"] for result in step.get("tool_results", [])]
        self.assertEqual(calls[0]["id"], "tu_1")
        self.assertEqual(results[0]["tool_call_id"], "tu_1")

    def test_validation_rejects_unmatched_tool_result(self):
        task = _claude_task()
        trace = Trajectory(task_id=task.id)
        trace.add("agent.start", agent="claude", provider="fixture", model="claude-sonnet-4-6")
        trace.add("agent.tool_call.observation", step=1, tool_name="read_file", tool_use_id="missing", observation={"ok": True})
        typed = build_typed_trajectory(trace, task=task).to_dict()

        result = validate_typed_trajectory(typed)
        self.assertFalse(result.ok)
        self.assertTrue(any("no matching tool call" in error for error in result.errors))

    def test_runner_writes_typed_trajectory_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture"
            fixture.mkdir()
            task = TaskSpec(
                id="typed-runner",
                title="typed runner",
                prompt="Do nothing.",
                workspace=WorkspaceSpec(fixture_path=str(fixture)),
                agent=AgentSpec(kind="noop"),
            )
            result = run_task(task, Settings.from_env(root=root))
            trajectory_path = result.run_dir / "trajectory.json"
            payload = json.loads(trajectory_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], TRAJECTORY_SCHEMA_VERSION)
        self.assertTrue(validate_typed_trajectory(payload).ok)

    def test_deterministic_agents_have_stable_semantic_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture"
            fixture.mkdir()
            task = TaskSpec(
                id="typed-stable",
                title="typed stable",
                prompt="Write answer.txt.",
                workspace=WorkspaceSpec(fixture_path=str(fixture)),
                agent=AgentSpec(kind="scripted", script=[{"action": "write_file", "path": "answer.txt", "content": "done\n"}]),
            )
            settings = Settings.from_env(root=root)
            first = run_task(task, settings)
            second = run_task(task, settings)
            first_payload = json.loads((first.run_dir / "trajectory.json").read_text(encoding="utf-8"))
            second_payload = json.loads((second.run_dir / "trajectory.json").read_text(encoding="utf-8"))

        self.assertEqual(trajectory_semantic_hash(first_payload), trajectory_semantic_hash(second_payload))


if __name__ == "__main__":
    unittest.main()
