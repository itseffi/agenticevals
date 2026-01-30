import tempfile
import unittest
from pathlib import Path

from agenticevals.agents.native_tools import GeminiNativeAgent, OpenAINativeAgent
from agenticevals.schema import AgentSpec, LimitsSpec, TaskSpec, WorkspaceSpec
from agenticevals.trace import Trajectory
from agenticevals.trajectory_export import build_typed_trajectory


class _FakeComputer:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def terminal(self, command: str, timeout: int = 60):
        raise AssertionError("terminal not used")

    def read_file(self, path: str) -> str:
        return (self.workspace / path).read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        target = self.workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _task(kind: str) -> TaskSpec:
    return TaskSpec(
        id=f"{kind}-native",
        title=f"{kind} native",
        prompt="Write answer.txt with done.",
        workspace=WorkspaceSpec(fixture_path="."),
        agent=AgentSpec(
            kind=kind,
            model="fixture",
            script=[
                {
                    "tool_calls": [{"id": "call_1", "name": "write_file", "input": {"path": "answer.txt", "content": "done\n"}}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
                {"content": "Created answer.txt.", "usage": {"input_tokens": 12, "output_tokens": 4}},
            ],
        ),
        limits=LimitsSpec(max_steps=5, max_minutes=1),
    )


class NativeAgentTests(unittest.TestCase):
    def test_openai_native_tool_loop_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            task = _task("openai")
            trace = Trajectory(task_id=task.id)
            result = OpenAINativeAgent().run(task, workspace, trace, timeout=30, computer=_FakeComputer(workspace))
            typed = build_typed_trajectory(trace, task=task)
            content = (workspace / "answer.txt").read_text(encoding="utf-8")

        self.assertTrue(result.ok)
        self.assertEqual(content, "done\n")
        self.assertEqual(typed.final_metrics.n_tool_calls, 1)
        self.assertEqual(typed.final_metrics.total_input_tokens, 22)
        self.assertEqual(typed.agent.provider, "fixture")

    def test_gemini_native_tool_loop_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            task = _task("gemini")
            trace = Trajectory(task_id=task.id)
            result = GeminiNativeAgent().run(task, workspace, trace, timeout=30, computer=_FakeComputer(workspace))
            typed = build_typed_trajectory(trace, task=task)
            content = (workspace / "answer.txt").read_text(encoding="utf-8")

        self.assertTrue(result.ok)
        self.assertEqual(content, "done\n")
        self.assertEqual(typed.final_metrics.n_tool_calls, 1)
        self.assertEqual(typed.final_metrics.total_output_tokens, 9)
        self.assertEqual(typed.agent.provider, "fixture")


if __name__ == "__main__":
    unittest.main()
