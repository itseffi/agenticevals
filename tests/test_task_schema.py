import unittest
import tempfile
import json
from pathlib import Path

from agenticevals.authoring import scaffold_task, validate_task_quality
from agenticevals.config import Settings
from agenticevals.canonical import normalize_run
from agenticevals.data_loop import write_dataset, write_improvement_loop
from agenticevals.environments import EnvironmentOptions, load_environment
from agenticevals.export import export_data, export_trajectories
from agenticevals.failures import classify_failure
from agenticevals.model_io import estimate_cost
from agenticevals.recompute import recompute_rewards
from agenticevals.runner import run_task
from agenticevals.scoring import score_dimensions
from agenticevals.schema import TaskSpec
from agenticevals.suites import load_suite, run_suite
from agenticevals.tool_parsers import parse_tool_calls
from agenticevals.trials import compute_pass_at_k, compute_pass_hat_k
from agenticevals.verify import verify_install
from agenticevals.viewer import write_review, write_viewer


class TaskSchemaTests(unittest.TestCase):
    def test_load_sample_task(self):
        task = TaskSpec.from_file(Path("configs/tasks/patch-python-bug.json"))
        self.assertEqual(task.id, "patch-python-bug")
        self.assertEqual(task.agent.kind, "scripted")
        self.assertTrue(task.resolve_fixture().exists())

    def test_load_declarative_tool_task(self):
        task = TaskSpec.from_file(Path("configs/tasks/mock-gmail-draft.json"))
        self.assertEqual(task.id, "mock-gmail-draft")
        self.assertEqual(task.tools[0].name, "gmail_list_messages")
        self.assertEqual(task.tool_endpoints[0].tool_name, "gmail_list_messages")
        self.assertEqual(task.expected_actions[0].action_key, "drafts")

    def test_load_command_agent_hidden_grader_task(self):
        task = TaskSpec.from_file(Path("configs/tasks/code-hidden-grader-command.json"))
        self.assertEqual(task.agent.kind, "command")
        self.assertIn("AGENTICEVALS_ROOT", task.agent.command)
        self.assertEqual(task.sandbox_grader_files, ["graders/code-hidden-grader"])

    def test_tool_call_parsers(self):
        calls = parse_tool_calls('<tool_call>{"tool_name":"write_file","arguments":{"path":"x","content":"y"}}</tool_call>')
        self.assertEqual(calls[0].tool_name, "write_file")
        self.assertEqual(calls[0].arguments["path"], "x")

    def test_model_loop_task_and_normalized_exports(self):
        task = TaskSpec.from_file(Path("configs/tasks/model-loop-write-file.json"))
        result = run_task(task, Settings.from_env())
        self.assertTrue(result.score.passed)
        normalized = normalize_run(result.run_dir)
        self.assertTrue(any(row.action_type == "agent.tool_call.parsed" for row in normalized))
        viewer = write_viewer(result.run_dir)
        self.assertTrue(viewer.exists())
        for kind in ["normalized", "training", "rl"]:
            exported = export_data(result.run_dir, kind=kind)
            self.assertTrue(exported.exists())
            self.assertGreater(len(exported.read_text(encoding="utf-8").splitlines()), 0)

    def test_load_suite(self):
        suite = load_suite(Path("configs/suites/core.json"))
        self.assertEqual(suite.id, "core")
        self.assertGreaterEqual(len(suite.tasks), 4)

        fuller = load_suite(Path("configs/suites/agentic-core.json"))
        self.assertEqual(fuller.id, "agentic-core")
        self.assertTrue(any(task.trials > 1 for task in fuller.tasks))

    def test_suite_run_checkpoint_and_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings.from_env(root=root)
            settings.ensure_dirs()
            suite_path = root / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "id": "unit-suite",
                        "title": "Unit suite",
                        "tasks": [
                            {"path": str(Path("configs/tasks/patch-python-bug.json").resolve())},
                            {"path": str(Path("configs/tasks/model-loop-write-file.json").resolve())},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = run_suite(suite_path, settings, workers=2)
            self.assertEqual(summary["passed"], summary["total"])
            run_dir = Path(summary["run_dir"])
            self.assertTrue((run_dir / "checkpoint.jsonl").exists())
            self.assertTrue((run_dir / "failures.json").exists())
            review = write_review(run_dir)
            self.assertTrue(review.exists())
            resumed = run_suite(suite_path, settings, workers=2, resume=run_dir)
            self.assertEqual(resumed["total"], summary["total"])

    def test_scaffolded_task_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings.from_env(root=root)
            settings.ensure_dirs()
            path = scaffold_task(settings, task_id="sample-author-task")
            self.assertTrue(path.exists())
            task = TaskSpec.from_file(path)
            self.assertTrue(task.resolve_fixture().exists())
            passed, issues = validate_task_quality(path, settings, run=True)
            self.assertTrue(passed, [issue.to_dict() for issue in issues])

    def test_load_environment(self):
        env = load_environment("examples.patch_python_bug_env:PatchPythonBugEnv", Settings.from_env())
        env.setup()
        item = env.get_next_item()
        self.assertIsNotNone(item)
        self.assertIn("divide", env.format_prompt(item))

    def test_environment_rollout(self):
        env = load_environment("examples.patch_python_bug_env:PatchPythonBugEnv", Settings.from_env())
        env.setup()
        item = env.get_next_item()
        result = env.rollout(item, EnvironmentOptions(agent="scripted", max_minutes=5, backend="local"))
        self.assertTrue(result.reward.passed)
        self.assertEqual(result.status, "passed")
        self.assertTrue((result.run_dir / "rollout.json").exists())

    def test_environment_evaluate_and_export(self):
        env = load_environment("examples.patch_python_bug_env:PatchPythonBugEnv", Settings.from_env())
        result = env.evaluate(EnvironmentOptions(agent="scripted", max_items=1, max_minutes=5, backend="local"))
        self.assertEqual(len(result.rollouts), 1)
        self.assertTrue((result.run_dir / "eval.json").exists())
        exported = export_trajectories(result.run_dir)
        self.assertTrue(exported.exists())
        self.assertEqual(len(exported.read_text(encoding="utf-8").splitlines()), 1)

        resumed = env.evaluate(EnvironmentOptions(agent="scripted", max_items=1, max_minutes=5, resume=result.run_dir, backend="local"))
        self.assertEqual(len(resumed.rollouts), 1)
        self.assertEqual(resumed.rollouts[0].item_id, result.rollouts[0].item_id)

    def test_browser_environment_rollout(self):
        env = load_environment("examples.browser_state_env:BrowserStateEnv", Settings.from_env())
        env.setup()
        item = env.get_next_item()
        result = env.rollout(item, EnvironmentOptions(agent="scripted", max_minutes=5, backend="local"))
        self.assertTrue(result.reward.passed)
        self.assertTrue((result.run_dir / "artifacts" / "browser" / "reward-browser-state.json").exists())

    def test_agent_smoke_scripted_and_noop(self):
        env = load_environment("examples.agent_smoke_env:AgentSmokeEnv", Settings.from_env())
        scripted = env.evaluate(EnvironmentOptions(agent="scripted", max_items=5, max_minutes=5, backend="local"))
        self.assertEqual(scripted.to_dict()["passed"], 5)

        env = load_environment("examples.agent_smoke_env:AgentSmokeEnv", Settings.from_env())
        noop = env.evaluate(EnvironmentOptions(agent="noop", max_items=5, max_minutes=5, backend="local"))
        self.assertLess(noop.to_dict()["passed"], 5)

    def test_tau_retail_smoke_environment(self):
        env = load_environment("examples.tau_retail_env:TauRetailEnv", Settings.from_env())
        result = env.evaluate(EnvironmentOptions(agent="scripted", max_items=3, max_minutes=5, backend="local"))
        self.assertEqual(result.to_dict()["passed"], 3)

    def test_data_exports(self):
        env = load_environment("examples.patch_python_bug_env:PatchPythonBugEnv", Settings.from_env())
        result = env.evaluate(EnvironmentOptions(agent="scripted", max_items=1, max_minutes=5, backend="local"))
        for kind in ["trajectories", "sharegpt", "actions", "rewards"]:
            exported = export_data(result.run_dir, kind=kind)
            self.assertTrue(exported.exists())
            self.assertGreater(len(exported.read_text(encoding="utf-8").splitlines()), 0)
        compressed = export_data(result.run_dir, kind="trajectories", compress=True)
        self.assertTrue(compressed.exists())

    def test_rl_dataset_recompute_and_improvement_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings.from_env(root=root)
            settings.ensure_dirs()
            suite_path = root / "suite.json"
            task_path = Path("configs/tasks/patch-python-bug.json").resolve()
            suite_path.write_text(
                json.dumps(
                    {
                        "id": "rl-suite",
                        "title": "RL suite",
                        "tasks": [
                            {"path": str(task_path), "agent": "scripted"},
                            {"path": str(task_path), "agent": "noop"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = run_suite(suite_path, settings, workers=1)
            self.assertEqual(summary["total"], 2)
            run_dir = Path(summary["run_dir"])

            rl_path = export_data(run_dir, kind="rl")
            rl_rows = [json.loads(line) for line in rl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["schema_version"] for row in rl_rows}, {"agenticevals.rl.v1"})
            self.assertTrue(any(not row["passed"] and row["hard_negative_tags"] for row in rl_rows))
            self.assertTrue(all("environment_hash" in row["metadata"] for row in rl_rows))

            preferences = export_data(run_dir, kind="preferences")
            preference_rows = [json.loads(line) for line in preferences.read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(preference_rows), 1)
            self.assertEqual(preference_rows[0]["schema_version"], "agenticevals.preference.v1")

            manifest = write_dataset(run_dir)
            self.assertEqual(manifest["schema_version"], "agenticevals.dataset-manifest.v1")
            self.assertTrue((run_dir / "dataset" / "DATASET.md").exists())

            failed_run = Path(next(row["run_dir"] for row in summary["tasks"] if not row["passed"]))
            recomputed = recompute_rewards(failed_run)
            self.assertTrue(recomputed["recomputed"])
            self.assertTrue(recomputed["matches_previous"])

            loop = write_improvement_loop(run_dir)
            self.assertGreaterEqual(loop["candidate_task_count"], 1)
            self.assertTrue((Path(loop["output_dir"]) / "candidate_tasks.jsonl").exists())

    def test_safety_policy_task(self):
        task = TaskSpec.from_file(Path("configs/tasks/safety-policy.json"))
        result = run_task(task, Settings.from_env())
        self.assertTrue(result.score.passed)

    def test_install_and_model_cost_helpers(self):
        install = verify_install()
        self.assertTrue(any(row["name"] == "module_cli" for row in install["checks"]))
        self.assertGreater(estimate_cost("gpt-4o-mini", {"input_tokens": 1000, "output_tokens": 1000}), 0.0)

    def test_failure_taxonomy(self):
        row = {
            "passed": False,
            "run_dir": "/path/that/does/not/exist",
            "summary": {"items": [{"name": "git_policy:required:src/app.py", "passed": False, "detail": "changed_files=[]"}]},
        }
        self.assertEqual(classify_failure(row).category, "policy_violation")

    def test_dimension_and_trial_scoring(self):
        task = TaskSpec.from_file(Path("configs/tasks/mock-gmail-draft.json"))
        dimensions = score_dimensions(
            audit_data={"gmail": {"drafts": [{"id": "draft_1"}], "sent": []}},
            dispatches=[],
            expected_actions=task.expected_actions,
            safety_checks=task.safety_checks,
            final_response="Saved the draft.",
        )
        self.assertTrue(dimensions.passed)
        self.assertEqual(dimensions.safety, 1.0)
        self.assertAlmostEqual(compute_pass_at_k([True, False, False], 1), 1 / 3)
        self.assertAlmostEqual(compute_pass_hat_k([True, False, False], 3), (1 / 3) ** 3)


if __name__ == "__main__":
    unittest.main()
