import tempfile
import unittest
from pathlib import Path

from agenticevals.backends.docker import _cached_image_tag
from agenticevals.sandbox import server


class SandboxPathTests(unittest.TestCase):
    def test_safe_path_rejects_workspace_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = server.WORKSPACE
            try:
                server.WORKSPACE = Path(tmp).resolve()
                with self.assertRaises(ValueError):
                    server._safe_path("../outside.txt")
            finally:
                server.WORKSPACE = original

    def test_safe_path_accepts_nested_workspace_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = server.WORKSPACE
            try:
                server.WORKSPACE = Path(tmp).resolve()
                resolved = server._safe_path("nested/file.txt")
            finally:
                server.WORKSPACE = original
        self.assertEqual(resolved, Path(tmp).resolve() / "nested" / "file.txt")

    def test_docker_cache_tag_changes_with_dockerfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            dockerfile = workspace / "Dockerfile"
            dockerfile.write_text("FROM python:3.12\n", encoding="utf-8")
            first = _cached_image_tag(workspace, dockerfile)
            dockerfile.write_text("FROM python:3.11\n", encoding="utf-8")
            second = _cached_image_tag(workspace, dockerfile)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
