from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from agenticevals.schema import ToolEndpointSpec, ToolSpec
from agenticevals.tools.types import ToolDispatchRecord, ToolResult
from agenticevals.trace import Trajectory


class ToolDispatchError(RuntimeError):
    pass


class ToolDispatcher:
    def __init__(self, tools: list[ToolSpec], endpoints: list[ToolEndpointSpec], trace: Trajectory | None = None, timeout: int = 30):
        self.tools = {tool.name: tool for tool in tools}
        self.endpoints = {endpoint.tool_name: endpoint for endpoint in endpoints}
        self.trace = trace
        self.timeout = timeout
        self.records: list[ToolDispatchRecord] = []

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        try:
            self._validate(tool_name, arguments)
            endpoint = self.endpoints[tool_name]
            status, body = self._request(endpoint, arguments)
            ok = 200 <= status < 400 and not (isinstance(body, dict) and body.get("error"))
            error = str(body.get("error", "")) if isinstance(body, dict) else ""
        except Exception as exc:
            status = 500
            body = {"error": str(exc)}
            ok = False
            error = str(exc)
        latency_ms = (time.monotonic() - started) * 1000
        record = ToolDispatchRecord(
            tool_name=tool_name,
            request=arguments,
            status=status,
            response=body,
            latency_ms=latency_ms,
            ok=ok,
            error=error,
        )
        self.records.append(record)
        if self.trace is not None:
            self.trace.add(
                "tool.dispatch",
                tool_name=tool_name,
                request=arguments,
                status=status,
                response=body,
                latency_ms=latency_ms,
                ok=ok,
                error=error,
            )
        return ToolResult(tool_name=tool_name, ok=ok, status=status, body=body, error=error)

    def _validate(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name not in self.tools:
            raise ToolDispatchError(f"unknown tool: {tool_name}")
        if tool_name not in self.endpoints:
            raise ToolDispatchError(f"tool has no endpoint: {tool_name}")
        schema = self.tools[tool_name].input_schema or {}
        required = schema.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ToolDispatchError(f"missing required tool arguments for {tool_name}: {', '.join(missing)}")
        properties = schema.get("properties", {})
        for key, value in arguments.items():
            expected = properties.get(key, {}).get("type")
            if expected and not _matches_json_type(value, expected):
                raise ToolDispatchError(f"invalid type for {tool_name}.{key}: expected {expected}")

    def _request(self, endpoint: ToolEndpointSpec, arguments: dict[str, Any]) -> tuple[int, Any]:
        payload = json.dumps(arguments).encode("utf-8")
        request = urllib.request.Request(
            endpoint.url,
            data=payload if endpoint.method.upper() != "GET" else None,
            headers={"Content-Type": "application/json"},
            method=endpoint.method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return response.status, json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {"error": raw}
            return exc.code, body


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True
