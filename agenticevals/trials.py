from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticevals.config import Settings
from agenticevals.runner import RunResult, run_task
from agenticevals.schema import TaskSpec


@dataclass(frozen=True)
class TrialSummary:
    run_dir: Path
    passed: bool
    points: float
    max_points: float
    task_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "passed": self.passed,
            "points": self.points,
            "max_points": self.max_points,
            "task_score": self.task_score,
        }


@dataclass(frozen=True)
class MultiTrialResult:
    task_id: str
    run_dir: Path
    trials: list[TrialSummary]

    def to_dict(self) -> dict[str, Any]:
        passed = [trial.passed for trial in self.trials]
        scores = [trial.task_score if trial.task_score is not None else (trial.points / trial.max_points if trial.max_points else 0.0) for trial in self.trials]
        return {
            "task_id": self.task_id,
            "run_dir": str(self.run_dir),
            "trials": [trial.to_dict() for trial in self.trials],
            "total_trials": len(self.trials),
            "passed_trials": sum(1 for item in passed if item),
            "pass_rate": sum(1 for item in passed if item) / len(passed) if passed else 0.0,
            "pass_at_1": compute_pass_at_k(passed, 1),
            "pass_hat_k": compute_pass_hat_k(passed, len(passed)),
            "pass_power_k": all(passed) if passed else False,
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
        }


def run_trials(task: TaskSpec, settings: Settings, *, agent_override: str | None, trials: int, use_sandbox_server: bool = False) -> MultiTrialResult:
    stamp = f"trials-{task.id}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = settings.runs_path / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    summaries: list[TrialSummary] = []
    for _ in range(trials):
        result = run_task(task, settings, agent_override=agent_override, use_sandbox_server=use_sandbox_server)
        summaries.append(_summary(result))
    multi = MultiTrialResult(task_id=task.id, run_dir=run_dir, trials=summaries)
    (run_dir / "trials.json").write_text(json.dumps(multi.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return multi


def compute_pass_at_k(passed: list[bool], k: int) -> float:
    n = len(passed)
    if n == 0 or k <= 0 or k > n:
        return 0.0
    c = sum(1 for item in passed if item)
    if c == 0:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_hat_k(passed: list[bool], k: int) -> float:
    """Unbiased pass^k: probability that a random k-subset of the n trials all pass.

    Uses the hypergeometric form C(c, k) / C(n, k) rather than the biased plug-in
    (c/n)^k. This mirrors the without-replacement estimator used by
    ``compute_pass_at_k`` and, at k == n, reduces to the observed all-passed
    indicator (1.0 iff every trial passed).
    """
    n = len(passed)
    if n == 0 or k <= 0 or k > n:
        return 0.0
    c = sum(1 for item in passed if item)
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def _summary(result: RunResult) -> TrialSummary:
    return TrialSummary(
        run_dir=result.run_dir,
        passed=result.score.passed,
        points=result.score.points,
        max_points=result.score.max_points,
        task_score=result.dimensions.task_score if result.dimensions is not None else None,
    )
