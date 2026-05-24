from __future__ import annotations

import json
from pathlib import Path
from typing import Type

from agenticevals.schema import VerifierSpec
from agenticevals.verifiers.base import BaseVerifier, VerifierContext, criterion
from agenticevals.verifiers.llm_rubric import LLMRubricVerifier
from agenticevals.verifiers.programmatic import ProgrammaticVerifier
from agenticevals.verifiers.schema import VerifierRunResult
from agenticevals.verifiers.state_check import StateCheckVerifier
from agenticevals.verifiers.tool_calls import ToolCallsVerifier
from agenticevals.verifiers.trajectory_check import TrajectoryCheckVerifier


VERIFIER_REGISTRY: dict[str, Type[BaseVerifier]] = {
    ProgrammaticVerifier.verifier_type: ProgrammaticVerifier,
    StateCheckVerifier.verifier_type: StateCheckVerifier,
    ToolCallsVerifier.verifier_type: ToolCallsVerifier,
    TrajectoryCheckVerifier.verifier_type: TrajectoryCheckVerifier,
    LLMRubricVerifier.verifier_type: LLMRubricVerifier,
}


def run_verifiers(context: VerifierContext) -> VerifierRunResult:
    criteria = []
    specs = context.task.verifiers or default_verifier_specs(context)
    for spec in specs:
        verifier_cls = VERIFIER_REGISTRY.get(spec.type)
        if verifier_cls is None:
            criteria.append(
                criterion(
                    name=spec.name or spec.type,
                    verifier_type=spec.type,
                    score=0.0,
                    weight=spec.weight,
                    passed=False,
                    detail=f"unknown verifier type: {spec.type}",
                    required=spec.required,
                    error=f"unknown verifier type: {spec.type}",
                )
            )
            continue
        try:
            criteria.extend(verifier_cls().verify(context, spec))
        except Exception as exc:
            criteria.append(
                criterion(
                    name=spec.name or spec.type,
                    verifier_type=spec.type,
                    score=0.0,
                    weight=spec.weight,
                    passed=False,
                    detail=f"verifier failed: {exc}",
                    required=spec.required,
                    error=str(exc),
                )
            )
    return VerifierRunResult(criteria=criteria)


def default_verifier_specs(context: VerifierContext) -> list[VerifierSpec]:
    task = context.task
    specs = [
        VerifierSpec(
            type="programmatic",
            name="command",
            weight=float(task.score.command_checks),
            config={"results": context.command_results},
        ),
        VerifierSpec(
            type="state_check",
            name="state",
            weight=1.0,
            config={
                "file_weight": float(task.score.file_checks),
                "browser_weight": float(task.score.browser_checks),
                "git_policy_weight": float(task.score.git_policy),
            },
        ),
    ]
    state_config = specs[-1].config
    if task.expected_actions:
        state_config["expected_actions_weight"] = float(task.score.expected_actions)
    if any(check.type == "audit_action_max_count" for check in task.safety_checks):
        state_config["audit_safety_weight"] = float(task.score.audit_safety)
    if task.tools or task.safety_checks:
        tool_config = {
            "dispatch_weight": float(task.score.tool_dispatch) if task.tools else 0.0,
            "argument_schema_weight": float(task.score.tool_argument) if task.tools else 0.0,
            "safety_weight": float(task.score.tool_safety) if any(check.type == "tool_not_called" for check in task.safety_checks) else 0.0,
        }
        specs.append(VerifierSpec(type="tool_calls", name="tool_calls", weight=1.0, config=tool_config))
    return specs


def write_reward_artifacts(run_dir: Path, result: VerifierRunResult) -> None:
    (run_dir / "reward.json").write_text(json.dumps(result.to_reward_dict(), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "reward-details.json").write_text(json.dumps(result.to_details_dict(), indent=2, sort_keys=True), encoding="utf-8")
