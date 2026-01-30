import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agenticevals.config import Settings, bundled_root
from agenticevals.schema import TaskSpec


class PackagingTests(unittest.TestCase):
    def test_default_config_discovery_uses_bundled_data_outside_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                with patch.dict(os.environ, {"AGENTICEVALS_CONFIG_ROOT": "", "AGENTICEVALS_TASK_CONFIG_DIR": ""}, clear=False):
                    os.environ.pop("AGENTICEVALS_CONFIG_ROOT", None)
                    os.environ.pop("AGENTICEVALS_TASK_CONFIG_DIR", None)
                    settings = Settings.from_env()
            finally:
                os.chdir(previous)

        self.assertEqual(settings.config_root, bundled_root() / "configs")
        task_path = settings.task_config_dir / "model-loop-write-file.json"
        task = TaskSpec.from_file(task_path)
        self.assertTrue(task.resolve_fixture().exists())


if __name__ == "__main__":
    unittest.main()
