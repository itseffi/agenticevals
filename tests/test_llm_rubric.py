import unittest
from pathlib import Path

from agenticevals.schema import AgentSpec, TaskSpec, VerifierSpec, WorkspaceSpec
from agenticevals.trace import Trajectory
from agenticevals.trajectory_export import build_typed_trajectory
from agenticevals.verifiers.base import VerifierContext
from agenticevals.verifiers.llm_rubric import (
    LLMRubricVerifier,
    _aggregate_judgments,
    _compact_transcript,
    _parse_json_object,
    _rubric_prompt,
)


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


class RubricTranscriptTests(unittest.TestCase):
    def test_default_truncation_is_500_chars(self):
        spec = VerifierSpec(type="llm_rubric", config={})
        transcript = _compact_transcript(_context("A" * 1000), spec)
        self.assertIn("A" * 500, transcript)
        self.assertNotIn("A" * 501, transcript)

    def test_transcript_max_chars_is_configurable(self):
        spec = VerifierSpec(type="llm_rubric", config={"transcript_max_chars": 10})
        transcript = _compact_transcript(_context("A" * 1000), spec)
        self.assertIn("A" * 10, transcript)
        self.assertNotIn("A" * 11, transcript)


class RubricFixtureGuardTests(unittest.TestCase):
    def test_fixture_score_is_flagged_as_not_a_real_judgment(self):
        verifier = LLMRubricVerifier()
        spec = VerifierSpec(type="llm_rubric", config={"fixture_score": 0.9, "threshold": 0.5})
        result = verifier.verify(_context(), spec)[0]
        # A hardcoded score must be distinguishable from a real LLM judgment so
        # it cannot silently masquerade as one in aggregated metrics.
        self.assertTrue(result.evidence.get("fixture"))


class RubricRepetitionTests(unittest.TestCase):
    def test_majority_vote_passes_when_most_judgments_pass(self):
        agg = _aggregate_judgments(
            [{"passed": True, "score": 0.9}, {"passed": True, "score": 0.8}, {"passed": False, "score": 0.2}], 0.5
        )
        self.assertTrue(agg["passed"])
        self.assertEqual(agg["votes"], [True, True, False])

    def test_majority_vote_fails_when_most_fail(self):
        agg = _aggregate_judgments([{"passed": False}, {"passed": False}, {"passed": True}], 0.5)
        self.assertFalse(agg["passed"])

    def test_even_split_breaks_tie_on_mean_score(self):
        agg = _aggregate_judgments([{"passed": True, "score": 0.4}, {"passed": False, "score": 0.4}], 0.5)
        self.assertFalse(agg["passed"])  # mean 0.4 < threshold 0.5

    def test_verify_runs_the_judge_repeatedly_and_aggregates(self):
        verifier = LLMRubricVerifier()
        calls = {"n": 0}
        seq = [{"passed": True}, {"passed": True}, {"passed": False}]

        def fake(context, spec, rep=0):
            i = calls["n"]
            calls["n"] += 1
            return seq[i]

        verifier._judge = fake  # type: ignore[method-assign]
        spec = VerifierSpec(type="llm_rubric", config={"repetitions": 3})
        result = verifier.verify(_context(), spec)[0]
        self.assertEqual(calls["n"], 3)
        self.assertTrue(result.passed)
        self.assertEqual(result.evidence["repetitions"], 3)


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
