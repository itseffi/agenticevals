import unittest

from agenticevals.trials import compute_pass_at_k, compute_pass_hat_k


class PassHatKTests(unittest.TestCase):
    def test_unbiased_estimator_uses_hypergeometric_form(self):
        # P(a random 2-subset of the 4 trials are all passes) = C(3,2)/C(4,2) = 0.5
        self.assertAlmostEqual(compute_pass_hat_k([True, True, True, False], 2), 0.5)

    def test_zero_when_fewer_successes_than_k(self):
        self.assertEqual(compute_pass_hat_k([True, False, False], 3), 0.0)

    def test_all_pass_is_one(self):
        self.assertEqual(compute_pass_hat_k([True, True], 2), 1.0)

    def test_reconciles_with_all_passed_at_k_equals_n(self):
        # At k == n, pass^k must equal the observed all-passed indicator.
        self.assertEqual(compute_pass_hat_k([True, True, False], 3), 0.0)
        self.assertEqual(compute_pass_hat_k([True, True, True], 3), 1.0)

    def test_guards(self):
        self.assertEqual(compute_pass_hat_k([], 1), 0.0)
        self.assertEqual(compute_pass_hat_k([True], 0), 0.0)
        self.assertEqual(compute_pass_hat_k([True, True], 3), 0.0)


class PassAtKTests(unittest.TestCase):
    def test_pass_at_1_is_success_fraction(self):
        self.assertAlmostEqual(compute_pass_at_k([True, False, False], 1), 1 / 3)


if __name__ == "__main__":
    unittest.main()
