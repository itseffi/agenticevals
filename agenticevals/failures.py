from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FailureClassification:
    category: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"category": self.category, "reason": self.reason}


def classify_failure(row: dict[str, Any]) -> FailureClassification:
    run_dir = Path(str(row.get("run_dir", "")))
    agent_error = _agent_error(run_dir)
    if agent_error:
        return FailureClassification("agent_runtime_error", agent_error)
    summary = row.get("summary", {})
    failed_items = [item for item in summary.get("items", []) if not item.get("passed", True)]
    for item in failed_items:
        name = str(item.get("name", ""))
        detail = str(item.get("detail", ""))
        if name.startswith("file:"):
            return FailureClassification("missing_or_invalid_artifact", f"{name}: {detail}")
        if name.startswith("git_policy:"):
            return FailureClassification("policy_violation", f"{name}: {detail}")
        if name.startswith("browser:"):
            return FailureClassification("browser_state_failure", f"{name}: {detail}")
        if name.startswith("command:"):
            return FailureClassification("command_check_failed", f"{name}: {detail}")
        if name.startswith("dimensions:"):
            return FailureClassification("dimension_failure", f"{name}: {detail}")
        if name.startswith("tool"):
            return FailureClassification("tool_failure", f"{name}: {detail}")
    if not bool(row.get("passed", True)):
        return FailureClassification("unknown", "task did not pass but no failed score item was recorded")
    return FailureClassification("passed", "task passed")


def aggregate_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("passed"):
            continue
        classification = classify_failure(row)
        item = {
            "task_id": row.get("task_id"),
            "agent": row.get("agent"),
            "run_dir": row.get("run_dir"),
            "score": row.get("score"),
            "reason": classification.reason,
        }
        clusters.setdefault(classification.category, []).append(item)
    return {
        "clusters": [
            {"label": label, "count": len(items), "items": items}
            for label, items in sorted(clusters.items(), key=lambda pair: (-len(pair[1]), pair[0]))
        ]
    }


def _agent_error(run_dir: Path) -> str | None:
    trace_path = run_dir / "trajectory.jsonl"
    if not trace_path.exists():
        return None
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "agent.result":
            continue
        data = event.get("data", {})
        if data.get("ok", True):
            return None
        message = str(data.get("final_message", "")).strip().replace("\n", " ")
        return message[:500] if message else "agent returned ok=false"
    return None
