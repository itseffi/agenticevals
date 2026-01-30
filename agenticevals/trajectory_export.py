from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from agenticevals.schema import TaskSpec
from agenticevals.trace import TraceEvent, Trajectory
from agenticevals.trajectory_schema import (
    AgentInfo,
    StepMetrics,
    TaskInfo,
    ToolCall,
    ToolResult,
    TrajectoryStep,
    TypedTrajectory,
    TRAJECTORY_SCHEMA_VERSION,
    canonical_json,
    compute_final_metrics,
    trajectory_semantic_hash,
)


def build_typed_trajectory(
    trace: Trajectory,
    *,
    task: TaskSpec | None = None,
    metadata: dict[str, Any] | None = None,
) -> TypedTrajectory:
    task_info = _task_info(trace, task)
    agent_info = _agent_info(trace, task)
    steps: list[TrajectoryStep] = []
    if task_info.prompt:
        steps.append(
            TrajectoryStep(
                index=1,
                source="user",
                kind="user_message",
                message=task_info.prompt,
            )
        )
    steps.extend(_steps_from_events(trace.events, start_index=len(steps) + 1))
    steps = [_renumber(step, index) for index, step in enumerate(steps, start=1)]
    final_metrics = compute_final_metrics(steps)
    trajectory = TypedTrajectory(
        schema_version=TRAJECTORY_SCHEMA_VERSION,
        run_id=trace.run_id,
        task=task_info,
        agent=agent_info,
        steps=steps,
        final_metrics=final_metrics,
        metadata=metadata or {},
    )
    return TypedTrajectory(
        schema_version=trajectory.schema_version,
        run_id=trajectory.run_id,
        task=trajectory.task,
        agent=trajectory.agent,
        steps=trajectory.steps,
        final_metrics=trajectory.final_metrics,
        metadata=trajectory.metadata,
        semantic_hash=trajectory_semantic_hash(trajectory),
    )


def write_typed_trajectory(
    trace: Trajectory,
    path: Path,
    *,
    task: TaskSpec | None = None,
    metadata: dict[str, Any] | None = None,
) -> TypedTrajectory:
    typed = build_typed_trajectory(trace, task=task, metadata=metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(typed.to_dict()), encoding="utf-8")
    return typed


def typed_trajectory_from_jsonl(
    path: Path,
    *,
    task: TaskSpec | None = None,
    run_id: str | None = None,
) -> TypedTrajectory:
    events = _load_jsonl_events(path)
    task_id = task.id if task is not None else _task_id_from_events(events, path.parent.name)
    trace = Trajectory(task_id=task_id, run_id=run_id or path.parent.name)
    trace.events.extend(events)
    return build_typed_trajectory(trace, task=task)


def _task_info(trace: Trajectory, task: TaskSpec | None) -> TaskInfo:
    if task is not None:
        return TaskInfo(id=task.id, title=task.title, prompt=task.prompt)
    title = ""
    prompt = ""
    for event in trace.events:
        if event.type == "run.start":
            title = str(event.data.get("title", ""))
            break
    return TaskInfo(id=trace.task_id, title=title, prompt=prompt)


def _agent_info(trace: Trajectory, task: TaskSpec | None) -> AgentInfo:
    kind = task.agent.kind if task is not None else "unknown"
    model = task.agent.model if task is not None else None
    provider = None
    metadata: dict[str, Any] = {}
    for event in trace.events:
        data = event.data
        if event.type == "agent.start":
            kind = str(data.get("agent") or kind)
            model = data.get("model") or model
            provider = data.get("provider") or provider
        elif event.type == "agent.result":
            result_meta = data.get("metadata")
            if isinstance(result_meta, dict):
                metadata.update(result_meta)
                model = result_meta.get("model") or model
                provider = result_meta.get("provider") or provider
    return AgentInfo(kind=str(kind), model=model, provider=provider, metadata=metadata)


def _steps_from_events(events: Iterable[TraceEvent], *, start_index: int) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    tool_ids_by_key: dict[tuple[str, str], deque[str]] = defaultdict(deque)
    generated_tool_count = 0
    saw_agent_finish = False
    for raw_index, event in enumerate(events):
        data = event.data
        if event.type in {"run.start", "environment.rollout.start"}:
            continue
        if event.type == "agent.start":
            steps.append(
                _event_step(
                    start_index + len(steps),
                    event,
                    raw_index,
                    source="system",
                    kind="agent_start",
                    extra=_select(data, "agent", "provider", "model", "max_steps", "tools"),
                )
            )
        elif event.type in {"agent.claude.turn", "agent.openai.turn", "agent.gemini.turn", "agent.model.response"}:
            steps.append(
                _event_step(
                    start_index + len(steps),
                    event,
                    raw_index,
                    source="agent",
                    kind="assistant_message",
                    message=str(data.get("text", "")),
                    metrics=_metrics_from_event(data),
                    stop_reason=data.get("stop_reason"),
                    extra=_select(data, "step", "provider", "model", "cached", "chars"),
                )
            )
        elif event.type == "agent.step":
            steps.append(
                _event_step(
                    start_index + len(steps),
                    event,
                    raw_index,
                    source="agent",
                    kind="agent_action",
                    extra=_select(data, "index", "action", "data"),
                )
            )
        elif event.type == "agent.tool_call.parsed":
            generated_tool_count += 1
            tool_id = str(data.get("tool_use_id") or f"tool-{generated_tool_count}")
            tool_name = str(data.get("tool_name", "unknown"))
            tool_ids_by_key[_tool_key(data, tool_name)].append(tool_id)
            steps.append(
                _event_step(
                    start_index + len(steps),
                    event,
                    raw_index,
                    source="agent",
                    kind="tool_call",
                    tool_calls=[ToolCall(id=tool_id, name=tool_name, arguments=dict(data.get("arguments", {})))],
                    extra=_select(data, "step"),
                )
            )
        elif event.type == "agent.tool_call.observation":
            tool_name = str(data.get("tool_name", "unknown"))
            tool_id = str(data.get("tool_use_id") or _pop_tool_id(tool_ids_by_key, data, tool_name) or f"tool-result-{raw_index}")
            observation = data.get("observation")
            steps.append(
                _event_step(
                    start_index + len(steps),
                    event,
                    raw_index,
                    source="tool",
                    kind="tool_result",
                    tool_results=[
                        ToolResult(
                            tool_call_id=tool_id,
                            name=tool_name,
                            content=observation,
                            is_error=not bool(observation.get("ok", True)) if isinstance(observation, dict) else False,
                        )
                    ],
                    observation=observation,
                    extra=_select(data, "step"),
                )
            )
        elif event.type == "tool.dispatch":
            generated_tool_count += 1
            tool_id = str(data.get("tool_use_id") or f"dispatch-{generated_tool_count}")
            tool_name = str(data.get("tool_name", "unknown"))
            response = data.get("response")
            steps.append(
                _event_step(
                    start_index + len(steps),
                    event,
                    raw_index,
                    source="tool",
                    kind="tool_dispatch",
                    tool_calls=[ToolCall(id=tool_id, name=tool_name, arguments=dict(data.get("request", {})))],
                    tool_results=[
                        ToolResult(
                            tool_call_id=tool_id,
                            name=tool_name,
                            content=response,
                            is_error=not bool(data.get("ok", False)),
                        )
                    ],
                    observation=response,
                    metrics=StepMetrics(latency_ms=round(float(data.get("latency_ms", 0.0) or 0.0), 3)),
                    status="ok" if data.get("ok") else "error",
                )
            )
        elif event.type == "agent.finish":
            saw_agent_finish = True
            steps.append(_final_step(start_index + len(steps), event, raw_index))
        elif event.type == "agent.result":
            if not saw_agent_finish:
                steps.append(_final_step(start_index + len(steps), event, raw_index))
        elif _is_observation_event(event.type):
            steps.append(
                _event_step(
                    start_index + len(steps),
                    event,
                    raw_index,
                    source=_source_for_event(event.type),
                    kind="observation",
                    observation=_observation_payload(data),
                    status=_status_for_event(data),
                    extra=_safe_extra_for_observation(data),
                )
            )
        elif event.type in {"score", "score.dimensions", "run.finish", "environment.rollout.finish"}:
            steps.append(
                _event_step(
                    start_index + len(steps),
                    event,
                    raw_index,
                    source="system",
                    kind="outcome",
                    status=_status_for_event(data),
                    observation=_observation_payload(data),
                )
            )
    return steps


def _event_step(
    index: int,
    event: TraceEvent,
    raw_index: int,
    *,
    source: str,
    kind: str,
    message: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    tool_results: list[ToolResult] | None = None,
    observation: Any = None,
    metrics: StepMetrics | None = None,
    status: str = "info",
    stop_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> TrajectoryStep:
    return TrajectoryStep(
        index=index,
        source=source,
        kind=kind,
        message=message,
        tool_calls=tool_calls or [],
        tool_results=tool_results or [],
        observation=observation,
        metrics=metrics or StepMetrics(),
        status=status,
        stop_reason=stop_reason,
        raw_event_type=event.type,
        raw_index=raw_index,
        extra=extra or {},
    )


def _final_step(index: int, event: TraceEvent, raw_index: int) -> TrajectoryStep:
    data = event.data
    return _event_step(
        index,
        event,
        raw_index,
        source="agent",
        kind="final_message",
        message=str(data.get("final_message", "")),
        status="ok" if data.get("ok", True) else "error",
        stop_reason=data.get("stop_reason"),
        extra=_select(data, "agent", "steps"),
    )


def _metrics_from_event(data: dict[str, Any]) -> StepMetrics:
    return StepMetrics(
        input_tokens=int(data.get("input_tokens", 0) or 0),
        output_tokens=int(data.get("output_tokens", 0) or 0),
        cache_creation_input_tokens=int(data.get("cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(data.get("cache_read_input_tokens", 0) or 0),
        cost_usd=round(float(data.get("cost_usd", 0.0) or 0.0), 6),
        latency_ms=round(float(data.get("latency_ms", 0.0) or 0.0), 3),
    )


def _tool_key(data: dict[str, Any], tool_name: str) -> tuple[str, str]:
    return (str(data.get("step", "")), tool_name)


def _pop_tool_id(ids: dict[tuple[str, str], deque[str]], data: dict[str, Any], tool_name: str) -> str | None:
    queue = ids.get(_tool_key(data, tool_name))
    if queue:
        return queue.popleft()
    return None


def _renumber(step: TrajectoryStep, index: int) -> TrajectoryStep:
    return TrajectoryStep(
        index=index,
        source=step.source,
        kind=step.kind,
        message=step.message,
        reasoning=step.reasoning,
        tool_calls=step.tool_calls,
        tool_results=step.tool_results,
        observation=step.observation,
        metrics=step.metrics,
        status=step.status,
        stop_reason=step.stop_reason,
        raw_event_type=step.raw_event_type,
        raw_index=step.raw_index,
        extra=step.extra,
    )


def _source_for_event(event_type: str) -> str:
    prefix = event_type.split(".", 1)[0]
    if prefix in {"agent", "computer", "tool", "service", "sandbox", "verify", "environment"}:
        return prefix
    if event_type.startswith("workspace."):
        return "workspace"
    if event_type.startswith("git."):
        return "git"
    return "system"


def _is_observation_event(event_type: str) -> bool:
    return event_type.startswith(
        (
            "computer.",
            "sandbox.",
            "verify.",
            "workspace.",
            "git.",
            "service.",
            "agent.shell",
            "agent.sandbox.",
        )
    )


def _status_for_event(data: dict[str, Any]) -> str:
    if "passed" in data:
        return "passed" if data.get("passed") else "failed"
    if "ok" in data:
        return "ok" if data.get("ok") else "error"
    if "returncode" in data:
        return "ok" if data.get("returncode") == 0 else "error"
    return "info"


def _observation_payload(data: dict[str, Any]) -> Any:
    if "observation" in data:
        return data["observation"]
    return _select(data, "passed", "detail", "returncode", "stdout", "stderr", "timed_out", "files", "changed_files", "points", "max_points")


def _safe_extra_for_observation(data: dict[str, Any]) -> dict[str, Any]:
    return _select(data, "name", "path", "command", "tool_name", "service")


def _select(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: data[key] for key in keys if key in data}


def _task_id_from_events(events: list[TraceEvent], fallback: str) -> str:
    for event in events:
        if event.type == "run.start":
            return str(event.data.get("task_id") or fallback)
        if event.type == "environment.rollout.start":
            return str(event.data.get("item_id") or fallback)
    return fallback


def _load_jsonl_events(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        events.append(TraceEvent(type=str(row.get("type", "unknown")), data=dict(row.get("data", {})), ts=float(row.get("ts", 0.0) or 0.0)))
    return events
