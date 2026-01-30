from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from agenticevals.agents.base import AdapterUnavailable, AgentRun, BaseAgent
from agenticevals.agents.builtin_tools import (
    BUILTIN_TOOL_NAMES,
    BUILTIN_TOOL_SCHEMAS,
    execute_builtin_tool,
)
from agenticevals.model_io import ANTHROPIC_MODEL_FAMILIES, AnthropicMessage, anthropic_messages
from agenticevals.schema import TaskSpec, ToolSpec
from agenticevals.trace import Trajectory


class ClaudeAgent(BaseAgent):
    """Anthropic Messages API agent with native tool-calling.

    Uses structured `tool_use` and `tool_result` blocks instead of regex-parsing
    free-form text. Records per-turn token usage, cache tokens, latency, and cost
    in the trajectory so cross-model comparisons have first-class accounting.

    Provider modes:
        - `anthropic` (default when ANTHROPIC_API_KEY is set): real API call.
        - `fixture`: replays task.agent.script entries, each describing the next
          assistant turn as {text, tool_calls, stop_reason, usage}. Used for
          deterministic tests and offline eval fixtures.
    """

    name = "claude"

    def __init__(self, fixture_provider: Callable[..., dict[str, Any]] | None = None):
        self._fixture_provider = fixture_provider

    def run(self, task: TaskSpec, workspace: Path, trace: Trajectory, timeout: int, computer: Any = None) -> AgentRun:
        provider = _resolve_provider(task, self._fixture_provider is not None)
        model = _resolve_model(task)
        tools = _build_tool_schemas(task.tools)
        system = _system_prompt(task)
        trace.add(
            "agent.start",
            agent=self.name,
            provider=provider,
            model=model,
            max_steps=task.limits.max_steps,
            tools=[tool["name"] for tool in tools],
        )

        messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
        totals = _UsageTotals()
        final_text = ""
        stop_reason = ""
        last_step = 0
        fixture_state = {"index": 0}

        for step in range(1, task.limits.max_steps + 1):
            last_step = step
            response = _request_turn(
                provider=provider,
                model=model,
                messages=messages,
                tools=tools,
                system=system,
                timeout=timeout,
                task=task,
                fixture_state=fixture_state,
                external_provider=self._fixture_provider,
            )
            totals.add(response)
            trace.add(
                "agent.claude.turn",
                step=step,
                provider=provider,
                model=model,
                stop_reason=response.stop_reason,
                text=response.text[-4000:],
                tool_calls=[{"name": call["name"], "input": call["input"]} for call in response.tool_calls],
                input_tokens=int(response.usage.get("input_tokens", 0) or 0),
                output_tokens=int(response.usage.get("output_tokens", 0) or 0),
                cache_creation_input_tokens=int(response.usage.get("cache_creation_input_tokens", 0) or 0),
                cache_read_input_tokens=int(response.usage.get("cache_read_input_tokens", 0) or 0),
                cost_usd=round(response.cost_usd, 6),
                latency_ms=round(response.latency_ms, 2),
                cached=response.cached,
            )

            if not response.tool_calls:
                final_text = response.text or ""
                stop_reason = response.stop_reason or "end_turn"
                trace.add(
                    "agent.finish",
                    agent=self.name,
                    steps=step,
                    final_message=final_text,
                    stop_reason=stop_reason,
                    total_input_tokens=totals.input_tokens,
                    total_output_tokens=totals.output_tokens,
                    total_cost_usd=round(totals.cost_usd, 6),
                )
                return AgentRun(
                    ok=True,
                    final_message=final_text,
                    metadata={
                        "agent": self.name,
                        "provider": provider,
                        "model": model,
                        "steps": step,
                        "stop_reason": stop_reason,
                        "usage": totals.to_dict(),
                    },
                )

            messages.append({"role": "assistant", "content": response.content_blocks})
            tool_result_blocks: list[dict[str, Any]] = []
            for call in response.tool_calls:
                trace.add(
                    "agent.tool_call.parsed",
                    step=step,
                    tool_name=call["name"],
                    arguments=call["input"],
                    tool_use_id=call["id"],
                )
                observation = execute_builtin_tool(call["name"], call["input"], computer)
                trace.add(
                    "agent.tool_call.observation",
                    step=step,
                    tool_name=call["name"],
                    observation=observation,
                    tool_use_id=call["id"],
                )
                is_error = not bool(observation.get("ok", True))
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps(observation, sort_keys=True, default=str),
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_result_blocks})

        stop_reason = "max_steps"
        trace.add(
            "agent.finish",
            agent=self.name,
            steps=last_step,
            final_message="",
            stop_reason=stop_reason,
            total_input_tokens=totals.input_tokens,
            total_output_tokens=totals.output_tokens,
            total_cost_usd=round(totals.cost_usd, 6),
        )
        return AgentRun(
            ok=False,
            final_message=f"claude loop exceeded max_steps={task.limits.max_steps}",
            metadata={
                "agent": self.name,
                "provider": provider,
                "model": model,
                "steps": last_step,
                "stop_reason": stop_reason,
                "usage": totals.to_dict(),
            },
        )


class _UsageTotals:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.cost_usd = 0.0
        self.turns = 0

    def add(self, response: AnthropicMessage) -> None:
        usage = response.usage or {}
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.cache_creation_input_tokens += int(usage.get("cache_creation_input_tokens", 0) or 0)
        self.cache_read_input_tokens += int(usage.get("cache_read_input_tokens", 0) or 0)
        self.cost_usd += float(response.cost_usd or 0.0)
        self.turns += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


def _resolve_provider(task: TaskSpec, has_external_fixture: bool) -> str:
    explicit = (task.agent.model or "").strip()
    if explicit == "fixture":
        return "fixture"
    if has_external_fixture:
        return "fixture"
    forced = os.environ.get("AGENTICEVALS_CLAUDE_PROVIDER")
    if forced:
        return forced
    return "anthropic"


def _resolve_model(task: TaskSpec) -> str:
    raw = (task.agent.model or os.environ.get("AGENTICEVALS_ANTHROPIC_MODEL") or "claude-sonnet-4-6").strip()
    if raw == "fixture":
        return os.environ.get("AGENTICEVALS_ANTHROPIC_MODEL", "claude-sonnet-4-6")
    return ANTHROPIC_MODEL_FAMILIES.get(raw, raw)


def _build_tool_schemas(task_tools: list[ToolSpec]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    declared_names: set[str] = set()
    for tool in task_tools:
        schemas.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema or {"type": "object", "properties": {}},
            }
        )
        declared_names.add(tool.name)
    for builtin in BUILTIN_TOOL_SCHEMAS:
        if builtin["name"] in declared_names:
            continue
        schemas.append(builtin)
    return schemas


def _system_prompt(task: TaskSpec) -> str:
    builtin_hint = (
        "You have a computer interface available via built-in tools: "
        + ", ".join(sorted(BUILTIN_TOOL_NAMES))
        + ". Use tools to inspect and modify the workspace. When the task is complete, "
        "respond with a final message and stop calling tools."
    )
    return builtin_hint


def _request_turn(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    system: str,
    timeout: int,
    task: TaskSpec,
    fixture_state: dict[str, Any],
    external_provider: Callable[..., dict[str, Any]] | None,
) -> AnthropicMessage:
    if provider == "fixture":
        provider_fn = external_provider or _script_fixture_provider(task, fixture_state)
        return anthropic_messages(
            model=model,
            messages=messages,
            tools=tools,
            system=system,
            timeout=timeout,
            fixture_provider=provider_fn,
        )
    if provider == "anthropic":
        return anthropic_messages(
            model=model,
            messages=messages,
            tools=tools,
            system=system,
            timeout=timeout,
        )
    raise AdapterUnavailable(f"claude provider is not supported: {provider}")


def _script_fixture_provider(task: TaskSpec, state: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    script = list(task.agent.script or [])

    def provider(messages: list[dict[str, Any]], tools: list[dict[str, Any]], system: str | None) -> dict[str, Any]:
        index = state["index"]
        state["index"] = index + 1
        if index >= len(script):
            return {
                "content": [{"type": "text", "text": "Done."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        entry = script[index]
        content: list[dict[str, Any]] = []
        text = str(entry.get("text") or entry.get("content") or "").strip()
        if text:
            content.append({"type": "text", "text": text})
        for call_index, call in enumerate(entry.get("tool_calls") or []):
            content.append(
                {
                    "type": "tool_use",
                    "id": str(call.get("id") or f"tu_{index}_{call_index}"),
                    "name": str(call["name"]),
                    "input": call.get("input") or {},
                }
            )
        stop_reason = entry.get("stop_reason")
        if not stop_reason:
            stop_reason = "tool_use" if any(block["type"] == "tool_use" for block in content) else "end_turn"
        usage = entry.get("usage") or {"input_tokens": 0, "output_tokens": 0}
        return {"content": content, "stop_reason": stop_reason, "usage": usage}

    return provider
