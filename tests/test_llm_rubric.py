import unittest
from pathlib import Path

from agenticevals.schema import AgentSpec, TaskSpec, VerifierSpec, WorkspaceSpec
from agenticevals.trace import Trajectory
from agenticevals.trajectory_export import build_typed_trajectory
from agenticevals.verifiers.base import VerifierContext
from agenticevals.verifiers.llm_rubric import LLMRubricVerifier, _parse_json_object, _rubric_prompt


def _context(final_response: str = "Done.") -> VerifierContext:
    task = TaskSpec(
        id="rubric",
        title="Rubric",
        prompt="Summarize the file.",
        workspace=WorkspaceSpec(fixture_path="."),
        agent=AgentSpec(kind="noop"),
    )
    trace = Trajectory(task_id=task.id)
    trace.add("agent.result", ok=True, final_message=final_response, metadata={})
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
        final_response=final_response,
    )


class RubricPromptTests(unittest.TestCase):
    def test_prompt_requests_reason_before_verdict(self):
        prompt = _rubric_prompt(_context(), VerifierSpec(type="llm_rubric", config={}))
        self.assertLess(prompt.index('"reason"'), prompt.index('"passed"'))

    def test_prompt_requests_binary_verdict(self):
        prompt = _rubric_prompt(_context(), VerifierSpec(type="llm_rubric", config={}))
        self.assertIn("boolean", prompt.lower())


class RubricParseTests(unittest.TestCase):
    def test_parses_prose_wrapped_json(self):
        text = 'Here is my verdict: {"reason": "ok", "passed": true} — thanks!'
        self.assertEqual(_parse_json_object(text), {"reason": "ok", "passed": True})

    def test_parses_fenced_json(self):
        text = '```json\n{"reason": "ok", "passed": false}\n```'
        self.assertEqual(_parse_json_object(text), {"reason": "ok", "passed": False})


class RubricAbstainTests(unittest.TestCase):
    def test_judge_failure_abstains_instead_of_failing_agent(self):
        verifier = LLMRubricVerifier()

        def _boom(context, spec):
            raise RuntimeError("api down")

        verifier._judge = _boom  # type: ignore[method-assign]
        spec = VerifierSpec(type="llm_rubric", name="rubric", required=True, config={})

        results = verifier.verify(_context(), spec)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result.error)
        # Abstain: must not drag down the weighted reward or fail a required run.
        self.assertEqual(result.weight, 0.0)
        self.assertFalse(result.required)


if __name__ == "__main__":
    unittest.main()
