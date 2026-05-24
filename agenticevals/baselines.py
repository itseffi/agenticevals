from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from agenticevals.config import Settings
from agenticevals.stats import bootstrap_ci, wilson_ci
from agenticevals.suites import run_suite


def eval_is_saturated(pass_rates: list[float]) -> bool:
    """True when every agent lands at the same extreme (all 0.0 or all 1.0).

    A saturated suite gives no discriminative signal — Phoenix recommends keeping
    capability evals in the 50-80% range so prompt/model changes remain visible.
    """
    if not pass_rates:
        return False
    unique = set(pass_rates)
    return unique == {0.0} or unique == {1.0}


def run_baselines(
    suite_path: Path,
    settings: Settings,
    *,
    agents: list[str],
    workers: int = 1,
    output: Path | None = None,
) -> dict[str, Any]:
    run_dir = output.expanduser().resolve() if output else settings.runs_path / f"baselines-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for agent in agents:
        summary = run_suite(suite_path, settings, agent_override=agent, workers=workers)
        passed = [bool(row["passed"]) for row in summary["tasks"]]
        scores = [float(row["score"]) for row in summary["tasks"]]
        rows.append(
            {
                "agent": agent,
                "suite_id": summary["id"],
                "suite_run_dir": summary["run_dir"],
                "tasks": summary["total"],
                "passed": summary["passed"],
                "pass_rate": summary["pass_rate"],
                "pass_rate_ci": wilson_ci(passed),
                "mean_score": summary["mean_score"],
                "mean_score_ci": bootstrap_ci(scores, seed=_seed(agent, "mean_score")),
            }
        )
    payload = {
        "schema_version": "agenticevals.baselines.v1",
        "suite": str(suite_path.expanduser().resolve()),
        "agents": agents,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": sorted(rows, key=lambda row: (-row["pass_rate"], -row["mean_score"], row["agent"])),
        "saturated": eval_is_saturated([row["pass_rate"] for row in rows]),
    }
    (run_dir / "baselines.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "RESULTS.md").write_text(_results_markdown(payload), encoding="utf-8")
    payload["run_dir"] = str(run_dir)
    return payload


def _results_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Baseline Results",
        "",
        f"Suite: `{payload['suite']}`",
        "",
        "| Agent | Tasks | Passed | Pass Rate | 95% CI | Mean Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        ci = row["pass_rate_ci"]
        lines.append(
            f"| {row['agent']} | {row['tasks']} | {row['passed']} | {row['pass_rate']:.3f} | "
            f"[{ci['low']:.3f}, {ci['high']:.3f}] | {row['mean_score']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _seed(*parts: str) -> int:
    return sum(ord(char) for part in parts for char in part)
