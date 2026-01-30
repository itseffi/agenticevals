from __future__ import annotations

from typing import Any

from agenticevals.schema import SafetyCheckSpec, ToolSpec, VerifierSpec
from agenticevals.trajectory_schema import ToolCall, TypedTrajectory
from agenticevals.verifiers.base import BaseVerifier, VerifierContext, criterion
from agenticevals.verifiers.schema import CriterionResult


class ToolCallsVerifier(BaseVerifier):
    verifier_type = "tool_calls"

    def verify(self, context: VerifierContext, spec: VerifierSpec) -> list[CriterionResult]:
        calls = _collect_tool_calls(context.trajectory)
        checks: list[CriterionResult] = []
        checks.extend(self._required_tools(calls, spec))
        checks.extend(self._forbidden_tools(calls, spec))
        checks.extend(self._required_sequence(calls, spec))
        checks.extend(self._argument_schema_checks(context, calls, spec))
        checks.extend(self._dispatch_checks(context, spec))
        checks.extend(self._safety_checks(context, spec))
        if not checks:
            return [
                criterion(
                    name=spec.name or "tool_calls",
                    verifier_type=self.verifier_type,
                    score=1.0,
                    weight=spec.weight,
                    passed=True,
                    detail="no tool-call checks configured",
                    required=spec.required,
                )
            ]
        return checks

    def _required_tools(self, calls: list[ToolCall], spec: VerifierSpec) -> list[CriterionResult]:
        required = spec.config.get("required_tools", [])
        if not required:
            return []
        rows: list[tuple[str, int]] = []
        for item in required:
            if isinstance(item, dict):
                rows.append((str(item["name"]), int(item.get("min_count", 1))))
            else:
                rows.append((str(item), 1))
        weight = float(spec.config.get("required_tools_weight", spec.weight))
        each = weight / len(rows)
        counts = _tool_counts(calls)
        return [
            criterion(
                name=f"tool_calls:required:{name}",
                verifier_type=self.verifier_type,
                score=1.0 if counts.get(name, 0) >= min_count else 0.0,
                weight=each,
                passed=counts.get(name, 0) >= min_count,
                detail=f"count={counts.get(name, 0)}, min_count={min_count}",
                required=spec.required,
                evidence={"tool": name, "count": counts.get(name, 0), "min_count": min_count},
            )
            for name, min_count in rows
        ]

    def _forbidden_tools(self, calls: list[ToolCall], spec: VerifierSpec) -> list[CriterionResult]:
        forbidden = [str(name) for name in spec.config.get("forbidden_tools", [])]
        if not forbidden:
            return []
        weight = float(spec.config.get("forbidden_tools_weight", spec.weight))
        each = weight / len(forbidden)
        counts = _tool_counts(calls)
        checks = []
        for name in forbidden:
            count = counts.get(name, 0)
            passed = count == 0
            checks.append(
                criterion(
                    name=f"tool_calls:forbidden:{name}",
                    verifier_type=self.verifier_type,
                    score=1.0 if passed else 0.0,
                    weight=each,
                    passed=passed,
                    detail=f"count={count}, max_count=0",
                    required=spec.required,
                    evidence={"tool": name, "count": count},
                )
            )
        return checks

    def _required_sequence(self, calls: list[ToolCall], spec: VerifierSpec) -> list[CriterionResult]:
        sequence = [str(name) for name in spec.config.get("required_sequence", [])]
        if not sequence:
            return []
        names = [call.name for call in calls]
        passed = _is_subsequence(sequence, names)
        return [
            criterion(
                name="tool_calls:required_sequence",
                verifier_type=self.verifier_type,
                score=1.0 if passed else 0.0,
                weight=float(spec.config.get("sequence_weight", spec.weight)),
                passed=passed,
                detail=f"expected={sequence}, observed={names}",
                required=spec.required,
                evidence={"expected": sequence, "observed": names},
            )
        ]

    def _argument_schema_checks(self, context: VerifierContext, calls: list[ToolCall], spec: VerifierSpec) -> list[CriterionResult]:
        weight = float(spec.config.get("argument_schema_weight", 0.0))
        if weight <= 0 or not context.task.tools:
            return []
        known_tools = {tool.name: tool for tool in context.task.tools}
        checked_calls = [call for call in calls if call.name in known_tools]
        if not checked_calls:
            return [
                criterion(
                    name="tool_calls:argument_schema",
                    verifier_type=self.verifier_type,
                    score=1.0,
                    weight=weight,
                    passed=True,
                    detail="no declared tool calls observed",
                    required=spec.required,
                )
            ]
        each = weight / len(checked_calls)
        return [_schema_criterion(call, known_tools[call.name], each, spec.required) for call in checked_calls]

    def _dispatch_checks(self, context: VerifierContext, spec: VerifierSpec) -> list[CriterionResult]:
        weight = float(spec.config.get("dispatch_weight", 0.0))
        if weight <= 0:
            return []
        if not context.dispatches:
            return [
                criterion(
                    name="tool_calls:dispatch_success",
                    verifier_type=self.verifier_type,
                    score=1.0,
                    weight=weight,
                    passed=True,
                    detail="no endpoint dispatches",
                    required=spec.required,
                )
            ]
        failures = [record for record in context.dispatches if not record.ok]
        success_ratio = (len(context.dispatches) - len(failures)) / len(context.dispatches)
        passed = not failures
        return [
            criterion(
                name="tool_calls:dispatch_success",
                verifier_type=self.verifier_type,
                score=success_ratio,
                weight=weight,
                passed=passed,
                detail=f"dispatches={len(context.dispatches)}, failures={len(failures)}",
                required=spec.required,
                evidence={
                    "failures": [
                        {"tool_name": item.tool_name, "status": item.status, "error": item.error}
                        for item in failures
                    ]
                },
            )
        ]

    def _safety_checks(self, context: VerifierContext, spec: VerifierSpec) -> list[CriterionResult]:
        weight = float(spec.config.get("safety_weight", 0.0))
        checks = [check for check in context.task.safety_checks if check.type == "tool_not_called" and check.tool_name]
        if weight <= 0 or not checks:
            return []
        each = weight / len(checks)
        counts = _tool_counts(_collect_tool_calls(context.trajectory))
        return [_tool_safety_criterion(check, counts.get(check.tool_name or "", 0), each, spec.required) for check in checks]


def _collect_tool_calls(trajectory: TypedTrajectory) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for step in trajectory.steps:
        calls.extend(step.tool_calls)
    return calls


def _tool_counts(calls: list[ToolCall]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for call in calls:
        counts[call.name] = counts.get(call.name, 0) + 1
    return counts


def _is_subsequence(expected: list[str], observed: list[str]) -> bool:
    cursor = 0
    for name in observed:
        if cursor < len(expected) and name == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def _schema_criterion(call: ToolCall, tool: ToolSpec, weight: float, required: bool) -> CriterionResult:
    errors = _validate_json_schema_subset(tool.input_schema or {}, call.arguments)
    passed = not errors
    return criterion(
        name=f"tool_calls:arguments:{call.name}:{call.id}",
        verifier_type=ToolCallsVerifier.verifier_type,
        score=1.0 if passed else 0.0,
        weight=weight,
        passed=passed,
        detail="ok" if passed else "; ".join(errors),
        required=required,
        evidence={"tool": call.name, "tool_call_id": call.id, "arguments": call.arguments},
    )


def _tool_safety_criterion(check: SafetyCheckSpec, count: int, weight: float, required: bool) -> CriterionResult:
    passed = count <= check.max_count
    return criterion(
        name=f"tool_calls:safety:{check.tool_name}",
        verifier_type=ToolCallsVerifier.verifier_type,
        score=1.0 if passed else 0.0,
        weight=weight,
        passed=passed,
        detail=f"count={count}, max_count={check.max_count}; {check.description}",
        required=required,
        evidence={"tool": check.tool_name, "count": count, "max_count": check.max_count},
    )


def _validate_json_schema_subset(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = schema.get("required", [])
    for key in required:
        if key not in arguments:
            errors.append(f"missing required argument: {key}")
    properties = schema.get("properties", {})
    for key, value in arguments.items():
        expected = properties.get(key, {}).get("type")
        if expected and not _matches_json_type(value, expected):
            errors.append(f"invalid type for {key}: expected {expected}")
    return errors


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
