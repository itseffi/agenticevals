from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Iterable

from agenticevals.canonical import normalize_run
from agenticevals.data_loop import build_preference_rows, build_rl_rows, collect_rollouts


def export_trajectories(run_dir: Path, output: Path | None = None) -> Path:
    return export_data(run_dir, output=output, kind="trajectories")


def export_data(run_dir: Path, output: Path | None = None, kind: str = "trajectories", compress: bool = False) -> Path:
    root = run_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    suffix = ".jsonl.gz" if compress else ".jsonl"
    target = output.expanduser().resolve() if output else root / f"{kind}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows_for_kind(root, kind)
    if compress:
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            _write_rows(handle, rows)
    else:
        with target.open("w", encoding="utf-8") as handle:
            _write_rows(handle, rows)
    return target


def _write_rows(handle, rows: Iterable[dict]) -> None:
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _rows_for_kind(root: Path, kind: str) -> list[dict]:
    rollouts = _load_rollouts(root)
    if kind == "trajectories":
        return rollouts
    if kind == "sharegpt":
        return [_sharegpt_row(root, rollout) for rollout in rollouts]
    if kind == "actions":
        return _action_rows(root, rollouts)
    if kind == "rewards":
        return _reward_rows(root, rollouts)
    if kind == "normalized":
        return [row.to_dict() for row in normalize_run(root)]
    if kind == "training":
        return _training_rows(root, rollouts)
    if kind == "rl":
        return build_rl_rows(root)
    if kind == "preferences":
        return build_preference_rows(root)
    raise ValueError(f"unknown export kind: {kind}")


def _load_rollouts(root: Path) -> list[dict]:
    return collect_rollouts(root)


def _trajectory_path_for_rollout(root: Path, rollout: dict) -> Path:
    source = Path(str(rollout["source_rollout_path"]))
    candidate = source.parent / "trajectory.jsonl"
    if candidate.exists():
        return candidate
    return root / "trajectory.jsonl"


def _load_events(root: Path, rollout: dict) -> list[dict]:
    path = _trajectory_path_for_rollout(root, rollout)
    if not path.exists():
        return []
    return _load_events_from_path(path)


def _load_events_from_path(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sharegpt_row(root: Path, rollout: dict) -> dict:
    events = _load_events(root, rollout)
    observations = []
    for event in events:
        if event.get("type", "").startswith("computer."):
            observations.append(json.dumps(event.get("data", {}), sort_keys=True))
    conversations = [{"from": "human", "value": str(rollout.get("prompt", ""))}]
    if observations:
        conversations.append({"from": "observation", "value": "\n".join(observations)})
    conversations.append({"from": "agent", "value": str(rollout.get("agent_result", {}).get("final_response", ""))})
    return {
        "id": f"{rollout.get('environment')}/{rollout.get('item_id')}/{rollout.get('run_id')}",
        "conversations": conversations,
        "reward": rollout.get("reward", {}),
        "source_rollout_path": rollout.get("source_rollout_path"),
    }


def _action_rows(root: Path, rollouts: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for rollout in rollouts:
        for index, event in enumerate(_load_events(root, rollout)):
            event_type = str(event.get("type", ""))
            if event_type.startswith("agent.") or event_type.startswith("computer."):
                rows.append(
                    {
                        "run_id": rollout.get("run_id"),
                        "environment": rollout.get("environment"),
                        "item_id": rollout.get("item_id"),
                        "index": index,
                        "event_type": event_type,
                        "data": event.get("data", {}),
                        "source_rollout_path": rollout.get("source_rollout_path"),
                    }
                )
    return rows


def _training_rows(root: Path, rollouts: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for rollout in rollouts:
        events = [row.to_dict() for row in normalize_run(_rollout_root(root, rollout))]
        rows.append(
            {
                "id": f"{rollout.get('environment')}/{rollout.get('item_id')}/{rollout.get('run_id')}",
                "prompt": rollout.get("prompt", ""),
                "final_response": rollout.get("agent_result", {}).get("final_response", ""),
                "reward": rollout.get("reward", {}),
                "events": events,
                "source_rollout_path": rollout.get("source_rollout_path"),
            }
        )
    return rows


def _rl_rows(root: Path, rollouts: list[dict]) -> list[dict]:
    return build_rl_rows(root)


def _messages_from_events(events: list[dict], rollout: dict) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": str(rollout.get("prompt", ""))}]
    for event in events:
        if event["action_type"] in {"agent.model.response", "agent.result", "agent.finish"}:
            text = event["data"].get("text") or event["data"].get("final_message")
            if text:
                messages.append({"role": "assistant", "content": str(text)})
        if event["action_type"].endswith(".observation"):
            messages.append({"role": "tool", "content": json.dumps(event["data"], sort_keys=True)})
    return messages


def _rollout_root(root: Path, rollout: dict) -> Path:
    source = Path(str(rollout.get("source_rollout_path", "")))
    if source.exists():
        return source.parent
    return root


def _reward_rows(root: Path, rollouts: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for rollout in rollouts:
        reward = rollout.get("reward", {})
        for component in reward.get("components", []):
            rows.append(
                {
                    "run_id": rollout.get("run_id"),
                    "environment": rollout.get("environment"),
                    "item_id": rollout.get("item_id"),
                    "component": component.get("name"),
                    "passed": component.get("passed"),
                    "value": component.get("value"),
                    "max_value": component.get("max_value"),
                    "detail": component.get("detail"),
                    "source_rollout_path": rollout.get("source_rollout_path"),
                }
            )
        dimensions_path = Path(str(rollout.get("source_rollout_path", ""))).parent / "dimensions.json"
        if dimensions_path.exists():
            dimensions = json.loads(dimensions_path.read_text(encoding="utf-8"))
            for name in ["completion", "robustness", "communication", "safety"]:
                value = float(dimensions.get(name, 0.0) or 0.0)
                # Graduated dimensions (e.g. communication, partial completion)
                # carry their fractional score for downstream RL/preference use;
                # `partial` flags credit between 0 and 1 that a binary `passed`
                # would otherwise discard.
                rows.append(
                    {
                        "run_id": rollout.get("run_id"),
                        "environment": rollout.get("environment"),
                        "item_id": rollout.get("item_id"),
                        "component": name,
                        "passed": math.isclose(value, 1.0) or value > 1.0,
                        "partial": 0.0 < value < 1.0,
                        "score": value,
                        "value": value,
                        "max_value": 1.0,
                        "detail": dimensions.get("details", {}).get(name),
                        "source_rollout_path": rollout.get("source_rollout_path"),
                    }
                )
    return rows
