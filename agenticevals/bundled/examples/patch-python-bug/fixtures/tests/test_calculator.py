import unittest

from src.calculator import divide


class CalculatorTests(unittest.TestCase):
    def test_divide_regular_numbers(self):
        self.assertEqual(divide(10, 2), 5)

    def test_divide_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            divide(10, 0)


if __name__ == "__main__":
    unittest.main()

