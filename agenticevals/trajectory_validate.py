from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticevals.trajectory_schema import TRAJECTORY_SCHEMA_VERSION, TypedTrajectory, trajectory_semantic_hash


@dataclass(frozen=True)
class TrajectoryValidationResult:
    ok: bool
    errors: list[str]

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise TrajectoryValidationError("; ".join(self.errors))


class TrajectoryValidationError(ValueError):
    pass


def validate_typed_trajectory(trajectory: TypedTrajectory | dict[str, Any]) -> TrajectoryValidationResult:
    data = trajectory.to_dict() if isinstance(trajectory, TypedTrajectory) else trajectory
    errors: list[str] = []
    _validate_root(data, errors)
    if errors:
        return TrajectoryValidationResult(ok=False, errors=errors)
    _validate_steps(data, errors)
    _validate_final_metrics(data, errors)
    _validate_semantic_hash(data, errors)
    return TrajectoryValidationResult(ok=not errors, errors=errors)


def validate_typed_trajectory_file(path: Path) -> TrajectoryValidationResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_typed_trajectory(payload)


def assert_valid_typed_trajectory(trajectory: TypedTrajectory | dict[str, Any]) -> None:
    validate_typed_trajectory(trajectory).raise_for_errors()


def _validate_root(data: dict[str, Any], errors: list[str]) -> None:
    if data.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TRAJECTORY_SCHEMA_VERSION}")
    if not isinstance(data.get("run_id"), str) or not data.get("run_id"):
        errors.append("run_id must be a non-empty string")
    if not isinstance(data.get("task"), dict):
        errors.append("task must be an object")
    elif not data["task"].get("id"):
        errors.append("task.id is required")
    if not isinstance(data.get("agent"), dict):
        errors.append("agent must be an object")
    elif not data["agent"].get("kind"):
        errors.append("agent.kind is required")
    if not isinstance(data.get("steps"), list):
        errors.append("steps must be a list")
    if not isinstance(data.get("final_metrics"), dict):
        errors.append("final_metrics must be an object")


def _validate_steps(data: dict[str, Any], errors: list[str]) -> None:
    seen_tool_calls: set[str] = set()
    allowed_sources = {"user", "agent", "tool", "computer", "environment", "verifier", "verify", "workspace", "git", "sandbox", "service", "system"}
    for expected_index, step in enumerate(data.get("steps", []), start=1):
        path = f"steps[{expected_index - 1}]"
        if step.get("index") != expected_index:
            errors.append(f"{path}.index must be {expected_index}")
        source = step.get("source")
        if source not in allowed_sources:
            errors.append(f"{path}.source is invalid: {source}")
        if not step.get("kind"):
            errors.append(f"{path}.kind is required")
        metrics = step.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append(f"{path}.metrics must be an object")
        for call_index, call in enumerate(step.get("tool_calls", [])):
            call_path = f"{path}.tool_calls[{call_index}]"
            call_id = call.get("id")
            if not call_id:
                errors.append(f"{call_path}.id is required")
                continue
            if call_id in seen_tool_calls:
                errors.append(f"{call_path}.id is duplicated: {call_id}")
            seen_tool_calls.add(str(call_id))
            if not call.get("name"):
                errors.append(f"{call_path}.name is required")
            if not isinstance(call.get("arguments", {}), dict):
                errors.append(f"{call_path}.arguments must be an object")
        for result_index, result in enumerate(step.get("tool_results", [])):
            result_path = f"{path}.tool_results[{result_index}]"
            tool_call_id = result.get("tool_call_id")
            if not tool_call_id:
                errors.append(f"{result_path}.tool_call_id is required")
            elif str(tool_call_id) not in seen_tool_calls:
                errors.append(f"{result_path}.tool_call_id has no matching tool call: {tool_call_id}")
            if not result.get("name"):
                errors.append(f"{result_path}.name is required")


def _validate_final_metrics(data: dict[str, Any], errors: list[str]) -> None:
    steps = data.get("steps", [])
    metrics = data.get("final_metrics", {})
    expected = {
        "n_steps": len(steps),
        "n_tool_calls": sum(len(step.get("tool_calls", [])) for step in steps),
        "n_tool_results": sum(len(step.get("tool_results", [])) for step in steps),
        "total_input_tokens": sum(int(step.get("metrics", {}).get("input_tokens", 0) or 0) for step in steps),
        "total_output_tokens": sum(int(step.get("metrics", {}).get("output_tokens", 0) or 0) for step in steps),
        "total_cache_creation_input_tokens": sum(int(step.get("metrics", {}).get("cache_creation_input_tokens", 0) or 0) for step in steps),
        "total_cache_read_input_tokens": sum(int(step.get("metrics", {}).get("cache_read_input_tokens", 0) or 0) for step in steps),
        "total_cost_usd": round(sum(float(step.get("metrics", {}).get("cost_usd", 0.0) or 0.0) for step in steps), 6),
        "total_latency_ms": round(sum(float(step.get("metrics", {}).get("latency_ms", 0.0) or 0.0) for step in steps), 3),
    }
    for key, expected_value in expected.items():
        actual = metrics.get(key)
        if isinstance(expected_value, float):
            if abs(float(actual or 0.0) - expected_value) > 1e-6:
                errors.append(f"final_metrics.{key} expected {expected_value}, got {actual}")
        elif actual != expected_value:
            errors.append(f"final_metrics.{key} expected {expected_value}, got {actual}")


def _validate_semantic_hash(data: dict[str, Any], errors: list[str]) -> None:
    semantic_hash = data.get("semantic_hash")
    if semantic_hash and semantic_hash != trajectory_semantic_hash(data):
        errors.append("semantic_hash does not match trajectory payload")
