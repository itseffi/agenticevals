from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ParsedToolCall:
    tool_name: str
    arguments: dict[str, Any]
    raw: str


class ToolCallParser(Protocol):
    name: str

    def parse(self, text: str) -> list[ParsedToolCall]:
        ...


class JsonToolCallParser:
    name = "json"

    def parse(self, text: str) -> list[ParsedToolCall]:
        calls: list[ParsedToolCall] = []
        for raw in _json_candidates(text):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            calls.extend(_calls_from_payload(payload, raw))
        return calls


class XmlToolCallParser:
    name = "xml"
    pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def parse(self, text: str) -> list[ParsedToolCall]:
        calls: list[ParsedToolCall] = []
        for match in self.pattern.finditer(text):
            raw = match.group(1)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            calls.extend(_calls_from_payload(payload, raw))
        return calls


class MarkdownToolCallParser:
    name = "markdown"
    pattern = re.compile(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", re.DOTALL)

    def parse(self, text: str) -> list[ParsedToolCall]:
        calls: list[ParsedToolCall] = []
        for match in self.pattern.finditer(text):
            raw = match.group(1)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            calls.extend(_calls_from_payload(payload, raw))
        return calls


PARSERS: dict[str, ToolCallParser] = {
    "json": JsonToolCallParser(),
    "xml": XmlToolCallParser(),
    "markdown": MarkdownToolCallParser(),
}


def parse_tool_calls(text: str, parser_names: list[str] | None = None) -> list[ParsedToolCall]:
    selected = parser_names or ["xml", "markdown", "json"]
    seen: set[tuple[str, str]] = set()
    calls: list[ParsedToolCall] = []
    for name in selected:
        parser = PARSERS[name]
        for call in parser.parse(text):
            key = (call.tool_name, json.dumps(call.arguments, sort_keys=True))
            if key not in seen:
                calls.append(call)
                seen.add(key)
    return calls


def _calls_from_payload(payload: Any, raw: str) -> list[ParsedToolCall]:
    if isinstance(payload, list):
        calls: list[ParsedToolCall] = []
        for item in payload:
            calls.extend(_calls_from_payload(item, raw))
        return calls
    if not isinstance(payload, dict):
        return []
    name = payload.get("tool_name") or payload.get("name") or payload.get("function")
    args = payload.get("arguments") or payload.get("args") or {}
    if isinstance(name, dict):
        args = name.get("arguments") or args
        name = name.get("name")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"value": args}
    if not isinstance(name, str) or not isinstance(args, dict):
        return []
    return [ParsedToolCall(tool_name=name, arguments=args, raw=raw)]


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]
    candidates.extend(match.group(0) for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL))
    return candidates
