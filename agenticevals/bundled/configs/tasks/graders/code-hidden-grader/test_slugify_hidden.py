import unittest

from src.text_tools import slugify


class SlugifyHiddenTests(unittest.TestCase):
    def test_removes_punctuation_and_collapses_separators(self):
        self.assertEqual(slugify("Agentic evals: tools, state, safety!"), "agentic-evals-tools-state-safety")

    def test_strips_edge_separators(self):
        self.assertEqual(slugify("  Ship-ready? yes.  "), "ship-ready-yes")


if __name__ == "__main__":
    unittest.main()
