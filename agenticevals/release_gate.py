from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_release_gate(
    *,
    baselines_path: Path,
    calibration_path: Path | None = None,
    min_agents: int = 3,
    min_kappa: float = 0.5,
    min_tpr: float = 0.7,
    min_tnr: float = 0.7,
    min_n: int = 50,
) -> dict[str, Any]:
    baselines = _read_json(baselines_path)
    rows = baselines.get("rows", [])
    unique_agents = {str(row.get("agent", "")) for row in rows}
    checks = []
    checks.append(_check("baseline_agent_count", len(unique_agents) >= min_agents, f"agents={len(unique_agents)}, required={min_agents}"))
    checks.append(_check("baseline_has_ci", all("pass_rate_ci" in row for row in rows), "all baseline rows include pass_rate_ci"))
    checks.append(_check("baseline_has_dataset", bool(baselines.get("suite")), f"suite={baselines.get('suite', '')}"))
    if calibration_path is not None:
        calibration = _read_json(calibration_path)
        kappa = float(calibration.get("kappa", 0.0) or 0.0)
        checks.append(_check("judge_kappa", kappa >= min_kappa, f"kappa={kappa:.3f}, required={min_kappa:.3f}"))
        # Binary calibrations report per-class rates; a judge can clear kappa
        # while being lenient on one class, so gate on TPR and TNR when present.
        if "tpr" in calibration:
            tpr = float(calibration.get("tpr", 0.0) or 0.0)
            checks.append(_check("judge_tpr", tpr >= min_tpr, f"tpr={tpr:.3f}, required={min_tpr:.3f}"))
        if "tnr" in calibration:
            tnr = float(calibration.get("tnr", 0.0) or 0.0)
            checks.append(_check("judge_tnr", tnr >= min_tnr, f"tnr={tnr:.3f}, required={min_tnr:.3f}"))
        if "n" in calibration:
            # A high kappa on a handful of labels is not trustworthy; require a
            # minimum labeled-sample size before headlining judge scores.
            n = int(calibration.get("n", 0) or 0)
            checks.append(_check("judge_sample_size", n >= min_n, f"n={n}, required={min_n}"))
    else:
        checks.append(_check("judge_kappa", False, "missing calibration report"))
    return {
        "schema_version": "agenticevals.release-gate.v1",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
