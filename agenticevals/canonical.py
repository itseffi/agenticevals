from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CanonicalEvent:
    run_id: str
    task_id: str
    index: int
    timestamp: float
    actor: str
    action_type: str
    name: str
    status: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_run(run_dir: Path) -> list[CanonicalEvent]:
    root = run_dir.expanduser().resolve()
    trajectory = root / "trajectory.jsonl"
    if not trajectory.exists():
        rollout = root / "rollout.json"
        if rollout.exists():
            trajectory = root / "trajectory.jsonl"
    if not trajectory.exists():
        return []
    events = _load_events(trajectory)
    task_id = root.name
    run_id = root.name
    for event in events:
        if event.get("type") in {"run.start", "environment.rollout.start"}:
            data = event.get("data", {})
            task_id = str(data.get("task_id") or data.get("item_id") or task_id)
            run_id = str(data.get("run_id") or run_id)
            break
    return [normalize_event(event, index=i, run_id=run_id, task_id=task_id) for i, event in enumerate(events)]


def normalize_event(event: dict[str, Any], *, index: int, run_id: str, task_id: str) -> CanonicalEvent:
    event_type = str(event.get("type", "unknown"))
    data = event.get("data", {}) if isinstance(event.get("data"), dict) else {"value": event.get("data")}
    actor = _actor_for(event_type)
    status = _status_for(event_type, data)
    name = _name_for(event_type, data)
    return CanonicalEvent(
        run_id=run_id,
        task_id=task_id,
        index=index,
        timestamp=float(event.get("ts", 0.0) or 0.0),
        actor=actor,
        action_type=event_type,
        name=name,
        status=status,
        summary=_summary(event_type, data),
        data=data,
    )


def write_normalized_jsonl(run_dir: Path, output: Path | None = None) -> Path:
    target = output.expanduser().resolve() if output else run_dir / "normalized.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = normalize_run(run_dir)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    return target


def _load_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _actor_for(event_type: str) -> str:
    prefix = event_type.split(".", 1)[0]
    if prefix in {"agent", "computer", "tool", "service", "sandbox", "verify", "score", "run", "environment"}:
        return prefix
    if event_type.startswith("workspace."):
        return "workspace"
    if event_type.startswith("git."):
        return "git"
    return "system"


def _status_for(event_type: str, data: dict[str, Any]) -> str:
    if "passed" in data:
        return "passed" if data.get("passed") else "failed"
    if "ok" in data:
        return "ok" if data.get("ok") else "error"
    if "returncode" in data:
        return "ok" if data.get("returncode") == 0 else "error"
    if event_type.endswith(".error"):
        return "error"
    return "info"


def _name_for(event_type: str, data: dict[str, Any]) -> str:
    for key in ("tool_name", "command", "name", "action", "path", "service"):
        if key in data:
            return str(data[key])
    return event_type


def _summary(event_type: str, data: dict[str, Any]) -> str:
    if "final_message" in data:
        return str(data["final_message"])[:240]
    if "command" in data:
        return str(data["command"])[:240]
    if "error" in data:
        return str(data["error"])[:240]
    if "detail" in data:
        return str(data["detail"])[:240]
    return event_type
