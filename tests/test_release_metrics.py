import json
import tempfile
import unittest
from pathlib import Path

from agenticevals.baselines import eval_is_saturated, run_baselines
from agenticevals.calibration import calibrate_judge_file, calibration_report, cohen_kappa
from agenticevals.config import Settings
from agenticevals.environment_baselines import run_environment_baselines
from agenticevals.release_gate import evaluate_release_gate
from agenticevals.review_cli import filtered_review_rows
from agenticevals.stats import bootstrap_ci
from agenticevals.suites import run_suite


class ReleaseMetricsTests(unittest.TestCase):
    def test_bootstrap_ci_contains_mean(self):
        ci = bootstrap_ci([True, False, True, True], samples=200, seed=1)
        self.assertAlmostEqual(ci["mean"], 0.75)
        self.assertLessEqual(ci["low"], ci["mean"])
        self.assertGreaterEqual(ci["high"], ci["mean"])

    def test_calibration_kappa_report(self):
        self.assertAlmostEqual(cohen_kappa(["pass", "fail"], ["pass", "fail"]), 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / "labels.jsonl"
            labels.write_text(
                "\n".join(
                    [
                        json.dumps({"human_passed": True, "judge_passed": True}),
                        json.dumps({"human_passed": False, "judge_passed": False}),
                        json.dumps({"human_passed": True, "judge_passed": False}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = calibrate_judge_file(labels)
            output_exists = Path(report["output"]).exists()
        self.assertEqual(report["n"], 3)
        self.assertTrue(output_exists)

    def test_saturation_detected_only_at_shared_extreme(self):
        self.assertTrue(eval_is_saturated([1.0, 1.0, 1.0]))  # everyone passes -> no signal
        self.assertTrue(eval_is_saturated([0.0, 0.0]))       # nobody passes -> no signal
        self.assertFalse(eval_is_saturated([1.0, 0.5]))      # discriminates between agents
        self.assertFalse(eval_is_saturated([0.5, 0.5]))      # mid-range, not an extreme
        self.assertFalse(eval_is_saturated([]))

    def test_calibration_reports_tpr_and_tnr_for_binary_labels(self):
        labels = [
            ("pass", "pass"),  # TP
            ("pass", "pass"),  # TP
            ("pass", "fail"),  # FN
            ("fail", "fail"),  # TN
            ("fail", "pass"),  # FP
        ]
        report = calibration_report(labels)
        # TPR = TP/(TP+FN) = 2/3 ; TNR = TN/(TN+FP) = 1/2
        self.assertAlmostEqual(report["tpr"], 2 / 3, places=6)
        self.assertEqual(report["tnr"], 0.5)

    def test_release_gate_fails_on_low_judge_tpr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baselines_path = root / "baselines.json"
            baselines_path.write_text(
                json.dumps(
                    {
                        "suite": "s",
                        "rows": [
                            {"agent": a, "pass_rate_ci": {}} for a in ("scripted", "noop", "model-loop")
                        ],
                    }
                ),
                encoding="utf-8",
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps({"kappa": 0.8, "tpr": 0.6, "tnr": 0.9}), encoding="utf-8"
            )
            gate = evaluate_release_gate(
                baselines_path=baselines_path, calibration_path=calibration_path
            )
        tpr_check = next(c for c in gate["checks"] if c["name"] == "judge_tpr")
        self.assertFalse(tpr_check["passed"])
        self.assertFalse(gate["passed"])

    def test_release_gate_fails_on_small_calibration_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baselines_path = root / "baselines.json"
            baselines_path.write_text(
                json.dumps(
                    {"suite": "s", "rows": [{"agent": a, "pass_rate_ci": {}} for a in ("scripted", "noop", "model-loop")]}
                ),
                encoding="utf-8",
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps({"kappa": 0.9, "n": 10}), encoding="utf-8")
            gate = evaluate_release_gate(baselines_path=baselines_path, calibration_path=calibration_path)
        size_check = next(c for c in gate["checks"] if c["name"] == "judge_sample_size")
        self.assertFalse(size_check["passed"])

    def test_baselines_release_gate_and_filtered_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings.from_env(root=root)
            settings.ensure_dirs()
            suite_path = root / "suite.json"
            task_path = Path("configs/tasks/patch-python-bug.json").resolve()
            suite_path.write_text(
                json.dumps(
                    {
                        "id": "baseline-suite",
                        "title": "Baseline suite",
                        "tasks": [
                            {"path": str(task_path), "agent": "scripted"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            suite_summary = run_suite(suite_path, settings, workers=1)
            rows = filtered_review_rows(Path(suite_summary["run_dir"]), ["status=passed"], limit=5)
            baselines = run_baselines(suite_path, settings, agents=["scripted", "noop", "model-loop"], workers=1)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps({"kappa": 0.75}), encoding="utf-8")
            gate = evaluate_release_gate(baselines_path=Path(baselines["run_dir"]) / "baselines.json", calibration_path=calibration_path)

        self.assertEqual(len(rows), 1)
        self.assertIn("pass_rate_ci", baselines["rows"][0])
        self.assertTrue(gate["passed"])

    def test_environment_baselines_report_pass_power_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings.from_env(root=Path(tmp))
            settings.ensure_dirs()
            result = run_environment_baselines(
                "examples.tau_retail_env:TauRetailEnv",
                settings,
                agents=["scripted"],
                max_items=2,
                trials=2,
            )

        row = result["rows"][0]
        self.assertEqual(row["items"], 2)
        self.assertEqual(row["trials"], 2)
        self.assertEqual(row["pass_at_1"], 1.0)
        self.assertEqual(row["pass_power_k"], 1.0)
        self.assertIn("cost_per_success_usd", row)


if __name__ == "__main__":
    unittest.main()
