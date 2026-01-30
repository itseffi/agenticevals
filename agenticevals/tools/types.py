from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    ok: bool
    status: int
    body: Any
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "status": self.status,
            "body": self.body,
            "error": self.error,
        }


@dataclass(frozen=True)
class ToolDispatchRecord:
    tool_name: str
    request: dict[str, Any]
    status: int
    response: Any
    latency_ms: float
    ok: bool
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
