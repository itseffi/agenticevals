from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from agenticevals.canonical import normalize_run
from agenticevals.failures import classify_failure

RL_SCHEMA_VERSION = "agenticevals.rl.v1"
PREFERENCE_SCHEMA_VERSION = "agenticevals.preference.v1"
DATASET_MANIFEST_SCHEMA_VERSION = "agenticevals.dataset-manifest.v1"
IMPROVEMENT_LOOP_SCHEMA_VERSION = "agenticevals.improvement-loop.v1"


def collect_rollouts(run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir.expanduser().resolve()
    if (root / "suite.json").exists():
        return _collect_suite_rollouts(root)
    if (root / "trials.json").exists():
        return _collect_trial_rollouts(root)
    rollout_paths = sorted((root / "rollouts").glob("*/rollout.json"))
    if not rollout_paths and (root / "rollout.json").exists():
        rollout_paths = [root / "rollout.json"]
    if rollout_paths:
        return [_load_rollout_path(path, run_kind="environment") for path in rollout_paths]
    if (root / "trajectory.jsonl").exists():
        return [_task_run_rollout(root)]
    return []


def build_rl_rows(run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir.expanduser().resolve()
    rows = []
    for rollout in collect_rollouts(root):
        events = [row.to_dict() for row in normalize_run(_rollout_root(root, rollout))]
        reward = _reward_for_rollout(rollout)
        task_key = _task_key(rollout)
        rows.append(
            {
                "schema_version": RL_SCHEMA_VERSION,
                "trajectory_id": _trajectory_id(rollout),
                "task_key": task_key,
                "prompt": rollout.get("prompt", ""),
                "observations": _observation_events(events),
                "actions": _action_events(events),
                "messages": _messages_from_events(events, rollout),
                "final_state": _final_state(rollout, events),
                "reward": reward.get("value", 0.0),
                "max_reward": reward.get("max_value", 0.0),
                "passed": bool(reward.get("passed", False)),
                "reward_components": reward.get("components", []),
                "failure_category": _failure_category(rollout),
                "hard_negative_tags": _hard_negative_tags(rollout, events),
                "metadata": _metadata(root, rollout, events),
            }
        )
    return rows


def build_preference_rows(run_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in build_rl_rows(run_dir):
        grouped.setdefault(str(row["task_key"]), []).append(row)
    pairs: list[dict[str, Any]] = []
    for task_key, rows in grouped.items():
        ordered = sorted(rows, key=_preference_rank, reverse=True)
        for chosen_index, chosen in enumerate(ordered):
            for rejected in ordered[chosen_index + 1 :]:
                if _preference_rank(chosen) == _preference_rank(rejected):
                    continue
                pairs.append(_preference_pair(task_key, chosen, rejected))
    return pairs


def build_hard_negative_rows(run_dir: Path) -> list[dict[str, Any]]:
    return [row for row in build_rl_rows(run_dir) if row["hard_negative_tags"] or not row["passed"]]


def write_dataset(run_dir: Path, output_dir: Path | None = None, *, name: str | None = None) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    target = (output_dir.expanduser().resolve() if output_dir else root / "dataset")
    target.mkdir(parents=True, exist_ok=True)
    files = {
        "rl": target / "rl.jsonl",
        "preferences": target / "preferences.jsonl",
        "hard_negatives": target / "hard_negatives.jsonl",
    }
    _write_jsonl(files["rl"], build_rl_rows(root))
    _write_jsonl(files["preferences"], build_preference_rows(root))
    _write_jsonl(files["hard_negatives"], build_hard_negative_rows(root))
    manifest = build_dataset_manifest(root, files, name=name)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (target / "DATASET.md").write_text(dataset_card_markdown(manifest), encoding="utf-8")
    return manifest


def build_dataset_manifest(run_dir: Path, files: dict[str, Path], *, name: str | None = None) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    rows = build_rl_rows(root)
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "name": name or root.name,
        "created_at": _now_iso(),
        "source_run_dir": str(root),
        "source_tasks": sorted({row["task_key"] for row in rows}),
        "agents": sorted({str(row["metadata"].get("agent", "")) for row in rows if row["metadata"].get("agent")}),
        "models": sorted({str(row["metadata"].get("model", "")) for row in rows if row["metadata"].get("model")}),
        "trajectory_count": len(rows),
        "preference_pair_count": len(build_preference_rows(root)),
        "hard_negative_count": len([row for row in rows if row["hard_negative_tags"] or not row["passed"]]),
        "scorer_version": "agenticevals.score.v1",
        "reward_definition": _reward_definition(rows),
        "filters": {"included": "all collected rollouts", "excluded": "none"},
        "schema_versions": {
            "rl": RL_SCHEMA_VERSION,
            "preference": PREFERENCE_SCHEMA_VERSION,
        },
        "hashes": {
            "source_run_dir": hash_path(root),
            **{key: hash_path(path) for key, path in files.items() if path.exists()},
        },
    }


def dataset_card_markdown(manifest: dict[str, Any]) -> str:
    tasks = "\n".join(f"- `{task}`" for task in manifest.get("source_tasks", [])) or "- none"
    agents = ", ".join(f"`{agent}`" for agent in manifest.get("agents", [])) or "none"
    models = ", ".join(f"`{model}`" for model in manifest.get("models", [])) or "none"
    return f"""# {manifest["name"]}

Schema: `{manifest["schema_version"]}`

Created: `{manifest["created_at"]}`

Source run: `{manifest["source_run_dir"]}`

## Contents

- Trajectories: {manifest["trajectory_count"]}
- Preference pairs: {manifest["preference_pair_count"]}
- Hard negatives: {manifest["hard_negative_count"]}
- Agents: {agents}
- Models: {models}

## Source Tasks

{tasks}

## Reward Definition

Scorer version: `{manifest["scorer_version"]}`

Components:

{_component_markdown(manifest.get("reward_definition", {}))}

## Hashes

```json
{json.dumps(manifest.get("hashes", {}), indent=2, sort_keys=True)}
```
"""


def write_improvement_loop(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    target = output_dir.expanduser().resolve() if output_dir else root / "improvement-loop"
    target.mkdir(parents=True, exist_ok=True)
    negatives = build_hard_negative_rows(root)
    clusters = _cluster_negatives(negatives)
    candidates = [_candidate_eval(row) for row in negatives]
    _write_jsonl(target / "hard_negatives.jsonl", negatives)
    _write_jsonl(target / "candidate_tasks.jsonl", candidates)
    payload = {
        "schema_version": IMPROVEMENT_LOOP_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "source_run_dir": str(root),
        "failure_clusters": clusters,
        "candidate_task_count": len(candidates),
        "next_steps": [
            "inspect hard_negatives.jsonl",
            "promote useful rows from candidate_tasks.jsonl into task configs",
            "rerun the affected suite",
            "export a new dataset from the rerun",
        ],
    }
    (target / "improvement_loop.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload | {"output_dir": str(target)}


def hash_path(path: Path) -> str:
    root = path.expanduser().resolve()
    digest = hashlib.sha256()
    if not root.exists():
        return "missing"
    if root.is_file():
        _hash_file(root, digest, root.name)
        return digest.hexdigest()
    for item in sorted(root.rglob("*")):
        if _skip_hash_path(item):
            continue
        rel = item.relative_to(root).as_posix()
        if item.is_dir():
            digest.update(f"dir:{rel}\n".encode())
        elif item.is_file():
            _hash_file(item, digest, rel)
    return digest.hexdigest()


def _collect_suite_rollouts(root: Path) -> list[dict[str, Any]]:
    data = _read_json(root / "suite.json", {})
    rows: list[dict[str, Any]] = []
    for task in data.get("tasks", []):
        child = Path(str(task.get("run_dir", "")))
        if not child.exists():
            continue
        for rollout in collect_rollouts(child):
            rollout.setdefault("suite_id", data.get("id"))
            rollout.setdefault("suite_run_dir", str(root))
            rollout.setdefault("task_path", task.get("task_path"))
            rows.append(rollout)
    return rows


def _collect_trial_rollouts(root: Path) -> list[dict[str, Any]]:
    data = _read_json(root / "trials.json", {})
    rows: list[dict[str, Any]] = []
    for index, trial in enumerate(data.get("trials", []), start=1):
        child = Path(str(trial.get("run_dir", "")))
        if not child.exists():
            continue
        for rollout in collect_rollouts(child):
            rollout["trial_index"] = index
            rollout["trial_group_id"] = data.get("task_id") or root.name
            rollout["trial_run_dir"] = str(root)
            rows.append(rollout)
    return rows


def _load_rollout_path(path: Path, *, run_kind: str) -> dict[str, Any]:
    data = _read_json(path, {})
    data["source_rollout_path"] = str(path)
    data["source_run_kind"] = run_kind
    return data


def _task_run_rollout(root: Path) -> dict[str, Any]:
    report = _read_json(root / "report.json", {})
    score = _read_json(root / "score.json", {})
    task_artifact = _read_json(root / "task.json", {})
    task = task_artifact.get("task", {}) if isinstance(task_artifact.get("task"), dict) else {}
    task_id = str(task.get("id") or report.get("task", {}).get("id") or root.name)
    agent = _agent_from_trace(root) or str(task.get("agent", {}).get("kind", "unknown"))
    reward = {
        "passed": score.get("passed", False),
        "value": score.get("points", 0.0),
        "max_value": score.get("max_points", 0.0),
        "components": [
            {
                "name": item.get("name"),
                "passed": item.get("passed"),
                "value": item.get("points"),
                "max_value": item.get("max_points"),
                "detail": item.get("detail"),
            }
            for item in score.get("items", [])
        ],
    }
    return {
        "run_id": report.get("run_id", root.name),
        "environment": "task",
        "item_id": task_id,
        "task_id": task_id,
        "task_path": task_artifact.get("source_path"),
        "agent": agent,
        "model": task.get("agent", {}).get("model"),
        "prompt": task.get("prompt", ""),
        "agent_result": {"final_response": _final_response_from_trace(root), "metadata": _agent_metadata_from_trace(root)},
        "reward": reward,
        "changed_files": report.get("changed_files", []),
        "artifacts": report.get("artifacts", {}),
        "source_rollout_path": str(root / "score.json"),
        "source_run_kind": "task",
    }


def _metadata(root: Path, rollout: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    rollout_root = _rollout_root(root, rollout)
    task_key = _task_key(rollout)
    return {
        "agent": rollout.get("agent"),
        "model": rollout.get("model") or rollout.get("agent_result", {}).get("metadata", {}).get("model"),
        "task": task_key,
        "task_id": rollout.get("task_id") or rollout.get("item_id"),
        "environment": rollout.get("environment"),
        "seed": rollout.get("seed") or rollout.get("agent_result", {}).get("metadata", {}).get("seed"),
        "environment_hash": hash_path(rollout_root / "workspace") if (rollout_root / "workspace").exists() else hash_path(rollout_root),
        "run_hash": hash_path(rollout_root),
        "task_hash": _task_hash(rollout_root, rollout),
        "source_run_kind": rollout.get("source_run_kind"),
        "source_rollout_path": rollout.get("source_rollout_path"),
        "trial_group_id": rollout.get("trial_group_id"),
        "trial_index": rollout.get("trial_index"),
        "event_count": len(events),
    }


def _task_hash(rollout_root: Path, rollout: dict[str, Any]) -> str:
    task_json = rollout_root / "task.json"
    if task_json.exists():
        return hash_path(task_json)
    item_json = rollout_root / "item.json"
    if item_json.exists():
        return hash_path(item_json)
    task_path = rollout.get("task_path")
    if task_path:
        return hash_path(Path(str(task_path)))
    return "unknown"


def _final_state(rollout: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    changed_files = rollout.get("changed_files")
    if changed_files is None:
        for event in events:
            if event["action_type"] == "git.changed_files":
                changed_files = event["data"].get("files", [])
                break
    return {
        "status": rollout.get("status") or ("passed" if _reward_for_rollout(rollout).get("passed") else "failed"),
        "final_response": rollout.get("agent_result", {}).get("final_response", ""),
        "changed_files": changed_files or [],
        "artifacts": rollout.get("artifacts", {}),
    }


def _failure_category(rollout: dict[str, Any]) -> str:
    reward = _reward_for_rollout(rollout)
    if reward.get("passed"):
        return "passed"
    row = {
        "passed": reward.get("passed", False),
        "run_dir": str(_rollout_root(Path("."), rollout)),
        "summary": {
            "items": [
                {
                    "name": component.get("name"),
                    "passed": component.get("passed"),
                    "detail": component.get("detail"),
                }
                for component in reward.get("components", [])
            ]
        },
    }
    return classify_failure(row).category


def _hard_negative_tags(rollout: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    tags: set[str] = set()
    reward = _reward_for_rollout(rollout)
    if not reward.get("passed", False):
        tags.add("failed_trajectory")
    for component in reward.get("components", []):
        if component.get("passed", True):
            continue
        name = str(component.get("name", ""))
        detail = str(component.get("detail", ""))
        if "forbidden" in name or "forbidden" in detail:
            tags.add("changed_forbidden_files")
        if "required" in name and "changed_files=[]" in detail:
            tags.add("missing_required_change")
        if name.startswith("tool") or "invalid type" in detail or "missing required tool" in detail:
            tags.add("wrong_tool_or_invalid_args")
        if name.startswith("browser"):
            tags.add("ignored_browser_state")
        if name.startswith("command"):
            tags.add("command_check_failed")
        if name.startswith("file"):
            tags.add("missing_or_invalid_artifact")
    if any(event["action_type"] == "agent.result" and event["data"].get("ok") is False for event in events):
        tags.add("agent_runtime_error")
    if any("exceeded max_steps" in str(event.get("summary", "")) for event in events):
        tags.add("looped_too_long")
    return sorted(tags)


def _preference_pair(task_key: str, chosen: dict[str, Any], rejected: dict[str, Any]) -> dict[str, Any]:
    reason = _preference_reason(chosen, rejected)
    return {
        "schema_version": PREFERENCE_SCHEMA_VERSION,
        "preference_id": hashlib.sha256(
            f"{chosen['trajectory_id']}::{rejected['trajectory_id']}".encode("utf-8")
        ).hexdigest()[:16],
        "task_key": task_key,
        "chosen_trajectory_id": chosen["trajectory_id"],
        "rejected_trajectory_id": rejected["trajectory_id"],
        "chosen": _preference_summary(chosen),
        "rejected": _preference_summary(rejected),
        "reason": reason,
    }


def _preference_rank(row: dict[str, Any]) -> tuple[float, float, float]:
    reward = float(row.get("reward") or 0.0)
    max_reward = float(row.get("max_reward") or 0.0)
    reward_rate = reward / max_reward if max_reward else 0.0
    unsafe_penalty = len([tag for tag in row.get("hard_negative_tags", []) if tag in {"changed_forbidden_files", "wrong_tool_or_invalid_args"}])
    pass_bonus = 1.0 if row.get("passed") else 0.0
    return (pass_bonus, reward_rate, -float(unsafe_penalty))


def _preference_reason(chosen: dict[str, Any], rejected: dict[str, Any]) -> str:
    if chosen["passed"] and not rejected["passed"]:
        return "chosen passed and rejected failed"
    chosen_tags = set(chosen.get("hard_negative_tags", []))
    rejected_tags = set(rejected.get("hard_negative_tags", []))
    if len(chosen_tags) < len(rejected_tags):
        return "chosen has fewer hard-negative tags"
    return "chosen has higher reward"


def _preference_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": row["trajectory_id"],
        "passed": row["passed"],
        "reward": row["reward"],
        "max_reward": row["max_reward"],
        "failure_category": row["failure_category"],
        "hard_negative_tags": row["hard_negative_tags"],
        "metadata": row["metadata"],
    }


def _observation_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["actor"] in {"computer", "tool", "service", "sandbox", "verify"}]


def _action_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["actor"] in {"agent", "computer", "tool", "sandbox"}]


def _messages_from_events(events: list[dict[str, Any]], rollout: dict[str, Any]) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": str(rollout.get("prompt", ""))}]
    for event in events:
        if event["action_type"] in {"agent.model.response", "agent.result", "agent.finish"}:
            text = event["data"].get("text") or event["data"].get("final_message")
            if text:
                _append_message(messages, {"role": "assistant", "content": str(text)})
        if event["action_type"].endswith(".observation"):
            _append_message(messages, {"role": "tool", "content": json.dumps(event["data"], sort_keys=True)})
    return messages


def _append_message(messages: list[dict[str, str]], message: dict[str, str]) -> None:
    if messages and messages[-1] == message:
        return
    messages.append(message)


def _reward_for_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
    reward = rollout.get("reward", {})
    return reward if isinstance(reward, dict) else {}


def _trajectory_id(rollout: dict[str, Any]) -> str:
    return f"{rollout.get('environment')}/{rollout.get('item_id')}/{rollout.get('run_id')}"


def _task_key(rollout: dict[str, Any]) -> str:
    return str(rollout.get("trial_group_id") or rollout.get("task_id") or rollout.get("item_id") or rollout.get("environment"))


def _rollout_root(root: Path, rollout: dict[str, Any]) -> Path:
    source = Path(str(rollout.get("source_rollout_path", "")))
    if source.exists():
        return source.parent
    run_dir = rollout.get("run_dir")
    if run_dir and Path(str(run_dir)).exists():
        return Path(str(run_dir))
    return root.expanduser().resolve()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _final_response_from_trace(root: Path) -> str:
    result = ""
    for event in _trace_events(root):
        if event.get("type") == "agent.result":
            result = str(event.get("data", {}).get("final_message", ""))
    return result


def _agent_from_trace(root: Path) -> str | None:
    for event in _trace_events(root):
        data = event.get("data", {})
        if event.get("type") == "agent.start" and data.get("agent"):
            return str(data["agent"])
    return None


def _agent_metadata_from_trace(root: Path) -> dict[str, Any]:
    for event in _trace_events(root):
        if event.get("type") == "agent.result":
            data = event.get("data", {})
            metadata = data.get("metadata", {})
            return dict(metadata) if isinstance(metadata, dict) else {}
    return {}


def _trace_events(root: Path) -> list[dict[str, Any]]:
    path = root / "trajectory.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reward_definition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    for row in rows:
        for component in row.get("reward_components", []):
            name = str(component.get("name", "unknown"))
            slot = components.setdefault(name, {"max_value": component.get("max_value"), "seen": 0})
            slot["seen"] += 1
    return {"components": components}


def _component_markdown(reward_definition: dict[str, Any]) -> str:
    components = reward_definition.get("components", {})
    if not components:
        return "- none"
    return "\n".join(f"- `{name}` max={data.get('max_value')} seen={data.get('seen')}" for name, data in sorted(components.items()))


def _cluster_negatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = row.get("failure_category") or "unknown"
        clusters.setdefault(str(label), []).append(row)
    return [
        {
            "label": label,
            "count": len(items),
            "trajectory_ids": [item["trajectory_id"] for item in items],
            "hard_negative_tags": sorted({tag for item in items for tag in item.get("hard_negative_tags", [])}),
        }
        for label, items in sorted(clusters.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    ]


def _candidate_eval(row: dict[str, Any]) -> dict[str, Any]:
    task_id = str(row["task_key"]).replace("/", "-")
    category = str(row.get("failure_category", "unknown")).replace("_", "-")
    return {
        "schema_version": "agenticevals.candidate-task.v1",
        "id": f"regression-{task_id}-{category}",
        "source_trajectory_id": row["trajectory_id"],
        "failure_category": row.get("failure_category"),
        "hard_negative_tags": row.get("hard_negative_tags", []),
        "prompt": row.get("prompt", ""),
        "suggested_checks": _suggested_checks(row),
    }


def _suggested_checks(row: dict[str, Any]) -> list[str]:
    checks = []
    for tag in row.get("hard_negative_tags", []):
        if tag == "changed_forbidden_files":
            checks.append("add forbidden_paths policy for protected files")
        elif tag == "wrong_tool_or_invalid_args":
            checks.append("add declarative tool schema and assert invalid dispatches fail")
        elif tag == "ignored_browser_state":
            checks.append("add browser-visible final-state check")
        elif tag == "missing_required_change":
            checks.append("add require_changed_files policy")
        elif tag == "looped_too_long":
            checks.append("lower max_steps or add stop-condition scoring")
        else:
            checks.append(f"add regression check for {tag}")
    return sorted(set(checks))


def _hash_file(path: Path, digest: "hashlib._Hash", rel: str) -> None:
    digest.update(f"file:{rel}\n".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def _skip_hash_path(path: Path) -> bool:
    parts = set(path.parts)
    generated_dirs = {"dataset", "improvement-loop"}
    return bool(parts & generated_dirs) or ".git" in parts or "__pycache__" in parts or path.suffix in {".pyc", ".pyo"}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
