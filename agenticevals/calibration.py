from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def build_calibration_set(run_dir: Path, *, sample_size: int = 100, seed: int = 0) -> list[dict[str, Any]]:
    """Sample real llm_rubric judge decisions from runs into a labeling template.

    Walks ``reward-details.json`` files, collects non-fixture llm_rubric criteria,
    and emits rows carrying the judge's verdict plus an empty ``human_passed`` for
    an expert to fill. The result is class-balanced (~50/50 on the judge verdict)
    so the labeled set is suitable for ``calibrate-judge``.
    """
    root = Path(run_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for details_path in sorted(root.rglob("reward-details.json")):
        data = json.loads(details_path.read_text(encoding="utf-8"))
        for crit in data.get("criteria", []):
            if crit.get("verifier_type") != "llm_rubric":
                continue
            if crit.get("evidence", {}).get("fixture"):
                continue
            rows.append(
                {
                    "source": str(details_path),
                    "criterion": crit.get("name", "llm_rubric"),
                    "judge_passed": bool(crit.get("passed")),
                    "judge_score": float(crit.get("score", 0.0) or 0.0),
                    "judge_reason": str(crit.get("detail", "")),
                    "human_passed": None,
                }
            )
    return _balance_and_sample(rows, sample_size, seed)


def write_calibration_set(run_dir: Path, *, output: Path, sample_size: int = 100, seed: int = 0) -> dict[str, Any]:
    rows = build_calibration_set(run_dir, sample_size=sample_size, seed=seed)
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    passed = sum(1 for row in rows if row["judge_passed"])
    return {"output": str(target), "n": len(rows), "judge_passed": passed, "judge_failed": len(rows) - passed}


def _balance_and_sample(rows: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    passed = [row for row in rows if row["judge_passed"]]
    failed = [row for row in rows if not row["judge_passed"]]
    rng.shuffle(passed)
    rng.shuffle(failed)
    half = max(1, sample_size // 2)
    combined = passed[:half] + failed[:half]
    rng.shuffle(combined)
    return combined


def calibrate_judge_file(path: Path, *, output: Path | None = None) -> dict[str, Any]:
    rows = _read_jsonl(path)
    labels = [_label_pair(row) for row in rows]
    result = calibration_report(labels, source=_display_path(path))
    target = output.expanduser().resolve() if output else path.expanduser().resolve().with_suffix(".calibration.json")
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["output"] = str(target)
    return result


def calibration_report(labels: list[tuple[str, str]], *, source: str = "") -> dict[str, Any]:
    human = [item[0] for item in labels]
    judge = [item[1] for item in labels]
    classes = sorted(set(human) | set(judge))
    matrix = {h: {j: 0 for j in classes} for h in classes}
    for h, j in labels:
        matrix[h][j] += 1
    kappa = cohen_kappa(human, judge)
    accuracy = sum(1 for h, j in labels if h == j) / len(labels) if labels else 0.0
    report = {
        "schema_version": "agenticevals.judge-calibration.v1",
        "source": source,
        "n": len(labels),
        "classes": classes,
        "accuracy": round(accuracy, 6),
        "kappa": round(kappa, 6),
        "confusion_matrix": matrix,
        "limitations": "v0.1 calibration uses 50 labeled examples; expand to 200+ before treating LLM judge scores as headline metrics.",
    }
    rates = _binary_rates(matrix, classes)
    if rates is not None:
        report["tpr"], report["tnr"] = rates
    return report


def _binary_rates(matrix: dict[str, dict[str, int]], classes: list[str]) -> tuple[float, float] | None:
    """True-positive and true-negative rate, treating "pass" as the positive class.

    Phoenix calibration wants both TPR and TNR above ~0.70; accuracy and kappa
    alone hide a judge that is lenient on one class. Only defined for binary
    pass/fail label sets.
    """
    if set(classes) != {"pass", "fail"}:
        return None
    tp = matrix["pass"]["pass"]
    fn = matrix["pass"]["fail"]
    tn = matrix["fail"]["fail"]
    fp = matrix["fail"]["pass"]
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return round(tpr, 6), round(tnr, 6)


def cohen_kappa(human: list[str], judge: list[str]) -> float:
    if len(human) != len(judge):
        raise ValueError("human and judge labels must have equal length")
    n = len(human)
    if n == 0:
        return 0.0
    labels = set(human) | set(judge)
    observed = sum(1 for a, b in zip(human, judge) if a == b) / n
    expected = 0.0
    for label in labels:
        p_human = sum(1 for item in human if item == label) / n
        p_judge = sum(1 for item in judge if item == label) / n
        expected += p_human * p_judge
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _label_pair(row: dict[str, Any]) -> tuple[str, str]:
    if "human_label" in row and "judge_label" in row:
        return str(row["human_label"]), str(row["judge_label"])
    if "human_passed" in row and "judge_passed" in row:
        return _bool_label(row["human_passed"]), _bool_label(row["judge_passed"])
    if "human_score" in row and "judge_score" in row:
        threshold = float(row.get("threshold", 0.5))
        return _bool_label(float(row["human_score"]) >= threshold), _bool_label(float(row["judge_score"]) >= threshold)
    raise ValueError("calibration row must contain human_label/judge_label, human_passed/judge_passed, or human_score/judge_score")


def _bool_label(value: Any) -> str:
    return "pass" if bool(value) else "fail"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines() if line.strip()]


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return resolved.name
