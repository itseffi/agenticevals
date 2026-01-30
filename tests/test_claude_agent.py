import json
import tempfile
import unittest
from pathlib import Path

from agenticevals.agents.claude_loop import ClaudeAgent
from agenticevals.schema import AgentSpec, LimitsSpec, TaskSpec, ToolSpec, WorkspaceSpec
from agenticevals.trace import Trajectory


class _FakeCommandResult:
    def __init__(self, ok=True, returncode=0, stdout="", stderr=""):
        self.ok = ok
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeComputer:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.terminal_calls: list[str] = []

    def terminal(self, command: str, timeout: int = 60) -> _FakeCommandResult:
        self.terminal_calls.append(command)
        return _FakeCommandResult(ok=True, returncode=0, stdout="ok", stderr="")

    def read_file(self, path: str) -> str:
        return (self.workspace / path).read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        target = self.workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _make_task(script: list[dict], *, max_steps: int = 10, tools: list[ToolSpec] | None = None) -> TaskSpec:
    return TaskSpec(
        id="claude-test",
        title="claude test",
        prompt="Do the thing.",
        workspace=WorkspaceSpec(fixture_path="."),
        agent=AgentSpec(kind="claude", model="fixture", script=script),
        limits=LimitsSpec(max_steps=max_steps, max_minutes=1),
        tools=list(tools or []),
    )


def _scripted_provider(script):
    state = {"index": 0}

    def provider(messages, tools, system):
        index = state["index"]
        state["index"] = index + 1
        if index >= len(script):
            return {"content": [{"type": "text", "text": "Done."}], "stop_reason": "end_turn", "usage": {}}
        return script[index]

    return provider


class ClaudeAgentTests(unittest.TestCase):
    def test_native_tool_use_then_final_text(self):
        """Loop terminates when assistant returns end_turn with no tool_use blocks."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "input.txt").write_text("hello\n", encoding="utf-8")
            script = [
                {
                    "content": [
                        {"type": "text", "text": "Reading the file."},
                        {"type": "tool_use", "id": "tu_1", "name": "read_file", "input": {"path": "input.txt"}},
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
                {
                    "content": [{"type": "text", "text": "The file says hello."}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 150, "output_tokens": 8},
                },
            ]
            task = _make_task(script=[])
            trace = Trajectory(task_id=task.id)
            agent = ClaudeAgent(fixture_provider=_scripted_provider(script))
            result = agent.run(task, workspace, trace, timeout=30, computer=_FakeComputer(workspace))

            self.assertTrue(result.ok)
            self.assertEqual(result.final_message, "The file says hello.")
            self.assertEqual(result.metadata["steps"], 2)
            self.assertEqual(result.metadata["stop_reason"], "end_turn")
            self.assertEqual(result.metadata["usage"]["input_tokens"], 250)
            self.assertEqual(result.metadata["usage"]["output_tokens"], 28)

    def test_tool_result_blocks_are_appended_to_messages(self):
        """After a tool_use turn, the next request should include a tool_result block keyed by tool_use_id."""
        captured_messages: list[list[dict]] = []
        script = [
            {
                "content": [{"type": "tool_use", "id": "tu_42", "name": "read_file", "input": {"path": "x.txt"}}],
                "stop_reason": "tool_use",
                "usage": {},
            },
            {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}},
        ]
        state = {"index": 0}

        def provider(messages, tools, system):
            captured_messages.append([{"role": m["role"], "content": m["content"]} for m in messages])
            index = state["index"]
            state["index"] = index + 1
            return script[index]

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "x.txt").write_text("yo", encoding="utf-8")
            task = _make_task(script=[])
            trace = Trajectory(task_id=task.id)
            agent = ClaudeAgent(fixture_provider=provider)
            agent.run(task, workspace, trace, timeout=30, computer=_FakeComputer(workspace))

        self.assertEqual(len(captured_messages), 2)
        second_request = captured_messages[1]
        self.assertEqual(second_request[-1]["role"], "user")
        tool_result_blocks = [block for block in second_request[-1]["content"] if isinstance(block, dict) and block.get("type") == "tool_result"]
        self.assertEqual(len(tool_result_blocks), 1)
        self.assertEqual(tool_result_blocks[0]["tool_use_id"], "tu_42")
        payload = json.loads(tool_result_blocks[0]["content"])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["content"], "yo")

    def test_max_steps_terminates_with_failure(self):
        """A loop that never stops calling tools fails cleanly at max_steps."""
        script = [
            {
                "content": [{"type": "tool_use", "id": f"tu_{i}", "name": "terminal", "input": {"command": "echo hi"}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            task = _make_task(script=[], max_steps=3)
            trace = Trajectory(task_id=task.id)
            agent = ClaudeAgent(fixture_provider=_scripted_provider(script))
            result = agent.run(task, workspace, trace, timeout=30, computer=_FakeComputer(workspace))

            self.assertFalse(result.ok)
            self.assertEqual(result.metadata["steps"], 3)
            self.assertEqual(result.metadata["stop_reason"], "max_steps")
            self.assertIn("max_steps=3", result.final_message)

    def test_cost_accounting_with_known_model(self):
        """Cost is accumulated using PRICES_PER_1K for the resolved model."""
        script = [
            {
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 1000},
            }
        ]
        task = TaskSpec(
            id="claude-cost",
            title="cost",
            prompt="hi",
            workspace=WorkspaceSpec(fixture_path="."),
            agent=AgentSpec(kind="claude", model="claude-sonnet-4-6", script=[]),
            limits=LimitsSpec(max_steps=2, max_minutes=1),
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace = Trajectory(task_id=task.id)
            agent = ClaudeAgent(fixture_provider=_scripted_provider(script))
            result = agent.run(task, Path(tmp), trace, timeout=30, computer=_FakeComputer(Path(tmp)))

        # sonnet pricing: $0.003/k input, $0.015/k output → 1k+1k = 0.018
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.metadata["usage"]["cost_usd"], 0.018, places=6)

    def test_cache_tokens_are_recorded(self):
        script = [
            {
                "content": [{"type": "text", "text": "cached run done"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 500,
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            task = _make_task(script=[])
            trace = Trajectory(task_id=task.id)
            agent = ClaudeAgent(fixture_provider=_scripted_provider(script))
            result = agent.run(task, Path(tmp), trace, timeout=30, computer=_FakeComputer(Path(tmp)))

        self.assertEqual(result.metadata["usage"]["cache_creation_input_tokens"], 200)
        self.assertEqual(result.metadata["usage"]["cache_read_input_tokens"], 500)

    def test_task_tools_are_merged_with_builtins(self):
        """Tools declared on the task should be merged with built-in computer tools when no name collision."""
        captured_tools: list[list[dict]] = []
        script = [{"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}}]
        state = {"index": 0}

        def provider(messages, tools, system):
            captured_tools.append(tools)
            return script[state["index"]]

        custom = ToolSpec(name="search", description="Search docs", input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]})
        task = _make_task(script=[], tools=[custom])
        with tempfile.TemporaryDirectory() as tmp:
            trace = Trajectory(task_id=task.id)
            agent = ClaudeAgent(fixture_provider=provider)
            agent.run(task, Path(tmp), trace, timeout=30, computer=_FakeComputer(Path(tmp)))

        names = [tool["name"] for tool in captured_tools[0]]
        self.assertIn("search", names)
        self.assertIn("terminal", names)
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)

    def test_trace_records_per_turn_usage_and_latency(self):
        script = [
            {"content": [{"type": "tool_use", "id": "tu_a", "name": "terminal", "input": {"command": "ls"}}], "stop_reason": "tool_use", "usage": {"input_tokens": 12, "output_tokens": 3}},
            {"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn", "usage": {"input_tokens": 30, "output_tokens": 7}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            task = _make_task(script=[])
            trace = Trajectory(task_id=task.id)
            agent = ClaudeAgent(fixture_provider=_scripted_provider(script))
            agent.run(task, Path(tmp), trace, timeout=30, computer=_FakeComputer(Path(tmp)))

            turns = [event for event in trace.events if event.type == "agent.claude.turn"]
            self.assertEqual(len(turns), 2)
            self.assertEqual(turns[0].data["input_tokens"], 12)
            self.assertEqual(turns[1].data["output_tokens"], 7)
            self.assertIn("latency_ms", turns[0].data)


if __name__ == "__main__":
    unittest.main()
