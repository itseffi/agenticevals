import unittest

from agenticevals.stats import wilson_ci


class WilsonCITests(unittest.TestCase):
    def test_full_pass_has_nondegenerate_lower_bound(self):
        ci = wilson_ci([True] * 5)
        self.assertEqual(ci["mean"], 1.0)
        self.assertEqual(ci["high"], 1.0)
        # Unlike the percentile bootstrap (which returns [1.0, 1.0]), Wilson
        # reflects the real uncertainty of a 5-sample run.
        self.assertLess(ci["low"], 1.0)
        self.assertAlmostEqual(ci["low"], 0.566, places=2)

    def test_zero_pass_has_nondegenerate_upper_bound(self):
        ci = wilson_ci([False] * 5)
        self.assertEqual(ci["mean"], 0.0)
        self.assertEqual(ci["low"], 0.0)
        self.assertGreater(ci["high"], 0.0)
        self.assertAlmostEqual(ci["high"], 0.434, places=2)

    def test_bounds_stay_within_unit_interval_and_bracket_mean(self):
        ci = wilson_ci([True, False, True, True])
        self.assertAlmostEqual(ci["mean"], 0.75)
        self.assertGreaterEqual(ci["low"], 0.0)
        self.assertLessEqual(ci["high"], 1.0)
        self.assertLessEqual(ci["low"], ci["mean"])
        self.assertGreaterEqual(ci["high"], ci["mean"])

    def test_empty_is_safe(self):
        ci = wilson_ci([])
        self.assertEqual((ci["mean"], ci["low"], ci["high"]), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
