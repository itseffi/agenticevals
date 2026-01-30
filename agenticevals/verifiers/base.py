from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticevals.computer.browser import BrowserCheckResult
from agenticevals.computer.files import FileCheckResult
from agenticevals.schema import TaskSpec, VerifierSpec
from agenticevals.tools.types import ToolDispatchRecord
from agenticevals.trace import Trajectory
from agenticevals.trajectory_schema import TypedTrajectory
from agenticevals.verifiers.schema import CriterionResult


@dataclass(frozen=True)
class VerifierContext:
    task: TaskSpec
    workspace: Path
    trajectory: TypedTrajectory
    raw_trace: Trajectory
    changed_files: list[str]
    command_results: list[tuple[str, bool, str]]
    file_results: list[FileCheckResult]
    browser_results: list[BrowserCheckResult]
    audit_data: dict[str, Any]
    dispatches: list[ToolDispatchRecord]
    final_response: str


class BaseVerifier:
    verifier_type = "base"

    def verify(self, context: VerifierContext, spec: VerifierSpec) -> list[CriterionResult]:
        raise NotImplementedError


def criterion(
    *,
    name: str,
    verifier_type: str,
    score: float,
    weight: float,
    passed: bool,
    detail: str = "",
    required: bool = True,
    deterministic: bool = True,
    evidence: dict[str, Any] | None = None,
    error: str = "",
) -> CriterionResult:
    clamped = max(0.0, min(1.0, float(score)))
    return CriterionResult(
        name=name,
        verifier_type=verifier_type,
        score=clamped,
        weight=float(weight),
        passed=bool(passed),
        detail=detail,
        required=required,
        deterministic=deterministic,
        evidence=evidence or {},
        error=error,
    )
