from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from agenticevals.agents.base import AdapterUnavailable, AgentRun, BaseAgent
from agenticevals.agents.builtin_tools import BUILTIN_TOOL_NAMES, BUILTIN_TOOL_SCHEMAS, execute_builtin_tool
from agenticevals.model_io import NativeToolResponse, gemini_generate_content, openai_responses_native
from agenticevals.schema import TaskSpec, ToolSpec
from agenticevals.trace import Trajectory


class OpenAINativeAgent(BaseAgent):
    name = "openai"

    def __init__(self, fixture_provider: Callable[..., dict[str, Any]] | None = None):
        self._fixture_provider = fixture_provider

    def run(self, task: TaskSpec, workspace: Path, trace: Trajectory, timeout: int, computer: Any = None) -> AgentRun:
        model = _openai_model(task)
        provider = "fixture" if _uses_fixture(task, self._fixture_provider) else "openai"
        tools = _openai_tools(task.tools)
        trace.add("agent.start", agent=self.name, provider=provider, model=model, max_steps=task.limits.max_steps, tools=[tool["name"] for tool in tools])
        input_items: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
        previous_response_id: str | None = None
        totals = _UsageTotals()
        fixture_state = {"index": 0}
        for step in range(1, task.limits.max_steps + 1):
            response = _openai_turn(
                provider=provider,
                model=model,
                input_items=input_items,
                tools=tools,
                previous_response_id=previous_response_id,
                timeout=timeout,
                task=task,
                fixture_state=fixture_state,
                external_provider=self._fixture_provider,
            )
            totals.add(response)
            previous_response_id = str((response.raw or {}).get("id") or previous_response_id or "")
            _trace_native_turn(trace, "agent.openai.turn", step, response)
            if not response.tool_calls:
                trace.add("agent.finish", agent=self.name, steps=step, final_message=response.text, stop_reason=response.stop_reason, **totals.trace_fields())
                return AgentRun(ok=True, final_message=response.text, metadata=_metadata(self.name, provider, model, step, totals))
            outputs = []
            for call in response.tool_calls:
                trace.add("agent.tool_call.parsed", step=step, tool_name=call["name"], arguments=call["input"], tool_use_id=call["id"])
                observation = execute_builtin_tool(call["name"], call["input"], computer)
                trace.add("agent.tool_call.observation", step=step, tool_name=call["name"], observation=observation, tool_use_id=call["id"])
                outputs.append({"type": "function_call_output", "call_id": call["id"], "output": json.dumps(observation, sort_keys=True, default=str)})
            input_items = outputs
        return _max_steps_result(self.name, provider, model, task, totals)


class GeminiNativeAgent(BaseAgent):
    name = "gemini"

    def __init__(self, fixture_provider: Callable[..., dict[str, Any]] | None = None):
        self._fixture_provider = fixture_provider

    def run(self, task: TaskSpec, workspace: Path, trace: Trajectory, timeout: int, computer: Any = None) -> AgentRun:
        model = _gemini_model(task)
        provider = "fixture" if _uses_fixture(task, self._fixture_provider) else "gemini"
        tools = _gemini_tools(task.tools)
        trace.add("agent.start", agent=self.name, provider=provider, model=model, max_steps=task.limits.max_steps, tools=[tool["name"] for tool in tools])
        contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": task.prompt}]}]
        totals = _UsageTotals()
        fixture_state = {"index": 0}
        for step in range(1, task.limits.max_steps + 1):
            response = _gemini_turn(
                provider=provider,
                model=model,
                contents=contents,
                tools=tools,
                timeout=timeout,
                task=task,
                fixture_state=fixture_state,
                external_provider=self._fixture_provider,
            )
            totals.add(response)
            _trace_native_turn(trace, "agent.gemini.turn", step, response)
            if not response.tool_calls:
                trace.add("agent.finish", agent=self.name, steps=step, final_message=response.text, stop_reason=response.stop_reason, **totals.trace_fields())
                return AgentRun(ok=True, final_message=response.text, metadata=_metadata(self.name, provider, model, step, totals))
            model_parts = [{"functionCall": {"name": call["name"], "args": call["input"]}} for call in response.tool_calls]
            contents.append({"role": "model", "parts": model_parts})
            response_parts = []
            for call in response.tool_calls:
                trace.add("agent.tool_call.parsed", step=step, tool_name=call["name"], arguments=call["input"], tool_use_id=call["id"])
                observation = execute_builtin_tool(call["name"], call["input"], computer)
                trace.add("agent.tool_call.observation", step=step, tool_name=call["name"], observation=observation, tool_use_id=call["id"])
                response_parts.append({"functionResponse": {"name": call["name"], "response": observation}})
            contents.append({"role": "user", "parts": response_parts})
        return _max_steps_result(self.name, provider, model, task, totals)


class _UsageTotals:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.latency_ms = 0.0
        self.turns = 0

    def add(self, response: NativeToolResponse) -> None:
        usage = response.usage or {}
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.cost_usd += float(response.cost_usd or 0.0)
        self.latency_ms += float(response.latency_ms or 0.0)
        self.turns += 1

    def trace_fields(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_cost_usd": round(self.cost_usd, 6),
            "total_latency_ms": round(self.latency_ms, 2),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"turns": self.turns, **self.trace_fields()}


def _openai_turn(
    *,
    provider: str,
    model: str,
    input_items: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    previous_response_id: str | None,
    timeout: int,
    task: TaskSpec,
    fixture_state: dict[str, Any],
    external_provider: Callable[..., dict[str, Any]] | None,
) -> NativeToolResponse:
    fixture_provider = None
    if provider == "fixture":
        fixture_provider = external_provider or _openai_fixture_provider(task, fixture_state)
    return openai_responses_native(
        model=model,
        input_items=input_items,
        tools=tools,
        previous_response_id=previous_response_id,
        timeout=timeout,
        fixture_provider=fixture_provider,
    )


def _gemini_turn(
    *,
    provider: str,
    model: str,
    contents: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout: int,
    task: TaskSpec,
    fixture_state: dict[str, Any],
    external_provider: Callable[..., dict[str, Any]] | None,
) -> NativeToolResponse:
    fixture_provider = None
    if provider == "fixture":
        fixture_provider = external_provider or _gemini_fixture_provider(task, fixture_state)
    return gemini_generate_content(model=model, contents=contents, tools=tools, timeout=timeout, fixture_provider=fixture_provider)


def _openai_fixture_provider(task: TaskSpec, state: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    script = list(task.agent.script or [])

    def provider(input_items: list[dict[str, Any]], tools: list[dict[str, Any]], previous_response_id: str | None) -> dict[str, Any]:
        index = state["index"]
        state["index"] = index + 1
        if index >= len(script):
            return {"id": f"fixture-{index}", "output": [{"type": "message", "content": [{"type": "output_text", "text": "Done."}]}], "usage": {}}
        entry = script[index]
        output = []
        if entry.get("content") or entry.get("text"):
            output.append({"type": "message", "content": [{"type": "output_text", "text": str(entry.get("content") or entry.get("text"))}]})
        for call_index, call in enumerate(entry.get("tool_calls") or []):
            output.append(
                {
                    "type": "function_call",
                    "call_id": str(call.get("id") or f"call_{index}_{call_index}"),
                    "name": str(call["name"]),
                    "arguments": json.dumps(call.get("input") or {}, sort_keys=True),
                }
            )
        return {"id": f"fixture-{index}", "output": output, "usage": entry.get("usage") or {}}

    return provider


def _gemini_fixture_provider(task: TaskSpec, state: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    script = list(task.agent.script or [])

    def provider(contents: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        index = state["index"]
        state["index"] = index + 1
        if index >= len(script):
            return {"candidates": [{"content": {"parts": [{"text": "Done."}]}}], "usageMetadata": {}}
        entry = script[index]
        parts = []
        if entry.get("content") or entry.get("text"):
            parts.append({"text": str(entry.get("content") or entry.get("text"))})
        for call in entry.get("tool_calls") or []:
            parts.append({"functionCall": {"name": str(call["name"]), "args": call.get("input") or {}, "id": str(call.get("id", ""))}})
        return {"candidates": [{"content": {"parts": parts}}], "usageMetadata": entry.get("usage") or {}}

    return provider


def _trace_native_turn(trace: Trajectory, event_type: str, step: int, response: NativeToolResponse) -> None:
    trace.add(
        event_type,
        step=step,
        provider=response.provider,
        model=response.model,
        stop_reason=response.stop_reason,
        text=response.text[-4000:],
        tool_calls=[{"name": call["name"], "input": call["input"]} for call in response.tool_calls],
        input_tokens=int(response.usage.get("input_tokens", 0) or 0),
        output_tokens=int(response.usage.get("output_tokens", 0) or 0),
        cost_usd=round(response.cost_usd, 6),
        latency_ms=round(response.latency_ms, 2),
        cached=response.cached,
    )


def _openai_tools(task_tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [{"type": "function", "name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]} for tool in _tool_schema_rows(task_tools)]


def _gemini_tools(task_tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [{"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]} for tool in _tool_schema_rows(task_tools)]


def _tool_schema_rows(task_tools: list[ToolSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    declared_names: set[str] = set()
    for tool in task_tools:
        rows.append({"name": tool.name, "description": tool.description, "input_schema": tool.input_schema or {"type": "object", "properties": {}}})
        declared_names.add(tool.name)
    for builtin in BUILTIN_TOOL_SCHEMAS:
        if builtin["name"] not in declared_names:
            rows.append(builtin)
    return rows


def _metadata(agent: str, provider: str, model: str, steps: int, totals: _UsageTotals) -> dict[str, Any]:
    return {"agent": agent, "provider": provider, "model": model, "steps": steps, "usage": totals.to_dict()}


def _max_steps_result(agent: str, provider: str, model: str, task: TaskSpec, totals: _UsageTotals) -> AgentRun:
    return AgentRun(
        ok=False,
        final_message=f"{agent} loop exceeded max_steps={task.limits.max_steps}",
        metadata={"agent": agent, "provider": provider, "model": model, "steps": task.limits.max_steps, "stop_reason": "max_steps", "usage": totals.to_dict()},
    )


def _uses_fixture(task: TaskSpec, fixture_provider: Callable[..., dict[str, Any]] | None) -> bool:
    return fixture_provider is not None or (task.agent.model or "").strip() == "fixture"


def _openai_model(task: TaskSpec) -> str:
    raw = (task.agent.model or os.environ.get("AGENTICEVALS_OPENAI_MODEL") or os.environ.get("AGENTICEVALS_DEFAULT_MODEL") or "gpt-4o-mini").strip()
    return os.environ.get("AGENTICEVALS_OPENAI_MODEL", "gpt-4o-mini") if raw == "fixture" else raw


def _gemini_model(task: TaskSpec) -> str:
    raw = (task.agent.model or os.environ.get("AGENTICEVALS_GEMINI_MODEL") or "gemini-2.5-flash").strip()
    return os.environ.get("AGENTICEVALS_GEMINI_MODEL", "gemini-2.5-flash") if raw == "fixture" else raw
