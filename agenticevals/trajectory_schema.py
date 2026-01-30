from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


TRAJECTORY_SCHEMA_VERSION = "agenticevals.trajectory.v1"


@dataclass(frozen=True)
class TaskInfo:
    id: str
    title: str = ""
    prompt: str = ""


@dataclass(frozen=True)
class AgentInfo:
    kind: str
    model: str | None = None
    provider: str | None = None
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: Any = None
    is_error: bool = False


@dataclass(frozen=True)
class StepMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class TrajectoryStep:
    index: int
    source: str
    kind: str
    message: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    observation: Any = None
    metrics: StepMetrics = field(default_factory=StepMetrics)
    status: str = "info"
    stop_reason: str | None = None
    raw_event_type: str | None = None
    raw_index: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalMetrics:
    n_steps: int
    n_tool_calls: int
    n_tool_results: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_input_tokens: int = 0
    total_cache_read_input_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0


@dataclass(frozen=True)
class TypedTrajectory:
    schema_version: str
    run_id: str
    task: TaskInfo
    agent: AgentInfo
    steps: list[TrajectoryStep]
    final_metrics: FinalMetrics
    metadata: dict[str, Any] = field(default_factory=dict)
    semantic_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("semantic_hash") is None:
            data.pop("semantic_hash", None)
        return _drop_none(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypedTrajectory":
        return cls(
            schema_version=str(data["schema_version"]),
            run_id=str(data["run_id"]),
            task=TaskInfo(**data["task"]),
            agent=AgentInfo(**data["agent"]),
            steps=[_step_from_dict(item) for item in data.get("steps", [])],
            final_metrics=FinalMetrics(**data["final_metrics"]),
            metadata=dict(data.get("metadata", {})),
            semantic_hash=data.get("semantic_hash"),
        )


def compute_final_metrics(steps: list[TrajectoryStep]) -> FinalMetrics:
    return FinalMetrics(
        n_steps=len(steps),
        n_tool_calls=sum(len(step.tool_calls) for step in steps),
        n_tool_results=sum(len(step.tool_results) for step in steps),
        total_input_tokens=sum(step.metrics.input_tokens for step in steps),
        total_output_tokens=sum(step.metrics.output_tokens for step in steps),
        total_cache_creation_input_tokens=sum(step.metrics.cache_creation_input_tokens for step in steps),
        total_cache_read_input_tokens=sum(step.metrics.cache_read_input_tokens for step in steps),
        total_cost_usd=round(sum(step.metrics.cost_usd for step in steps), 6),
        total_latency_ms=round(sum(step.metrics.latency_ms for step in steps), 3),
    )


def trajectory_semantic_hash(trajectory: TypedTrajectory | dict[str, Any]) -> str:
    payload = trajectory.to_dict() if isinstance(trajectory, TypedTrajectory) else dict(trajectory)
    normalized = _normalize_for_hash(payload)
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"


def _step_from_dict(data: dict[str, Any]) -> TrajectoryStep:
    return TrajectoryStep(
        index=int(data["index"]),
        source=str(data["source"]),
        kind=str(data["kind"]),
        message=data.get("message"),
        reasoning=data.get("reasoning"),
        tool_calls=[ToolCall(**item) for item in data.get("tool_calls", [])],
        tool_results=[ToolResult(**item) for item in data.get("tool_results", [])],
        observation=data.get("observation"),
        metrics=StepMetrics(**data.get("metrics", {})),
        status=str(data.get("status", "info")),
        stop_reason=data.get("stop_reason"),
        raw_event_type=data.get("raw_event_type"),
        raw_index=data.get("raw_index"),
        extra=dict(data.get("extra", {})),
    )


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_for_hash(item)
            for key, item in value.items()
            if key not in {"run_id", "semantic_hash", "raw_index"}
        }
    if isinstance(value, list):
        return [_normalize_for_hash(item) for item in value]
    return value
