import unittest

from agenticevals.scoring.dimensions import _score_communication, score_dimensions


def _dims(final_response: str):
    return score_dimensions(
        audit_data={},
        dispatches=[],
        expected_actions=[],
        safety_checks=[],
        final_response=final_response,
    )


class CommunicationTests(unittest.TestCase):
    def test_empty_response_scores_zero(self):
        score, _ = _score_communication("")
        self.assertEqual(score, 0.0)

    def test_thin_response_scores_partial(self):
        score, _ = _score_communication("ok")
        self.assertEqual(score, 0.5)

    def test_substantive_response_scores_full(self):
        score, _ = _score_communication("Saved the draft successfully and verified it.")
        self.assertEqual(score, 1.0)

    def test_communication_contributes_to_task_score(self):
        thin = _dims("ok")
        full = _dims("Saved the draft successfully and verified it.")
        # Everything else equal, a thin final message should score lower overall.
        self.assertLess(thin.task_score, full.task_score)


if __name__ == "__main__":
    unittest.main()
