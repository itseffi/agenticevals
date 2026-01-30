import unittest

from src.app import normalize_name


class AppTests(unittest.TestCase):
    def test_normalize_name(self):
        self.assertEqual(normalize_name(" Ada Lovelace "), "ada-lovelace")


if __name__ == "__main__":
    unittest.main()
