from __future__ import annotations

from agenticevals.schema import VerifierSpec
from agenticevals.verifiers.base import BaseVerifier, VerifierContext, criterion
from agenticevals.verifiers.schema import CriterionResult


class TrajectoryCheckVerifier(BaseVerifier):
    verifier_type = "trajectory_check"

    def verify(self, context: VerifierContext, spec: VerifierSpec) -> list[CriterionResult]:
        checks: list[CriterionResult] = []
        configured = False
        if "require_final_message" in spec.config:
            configured = True
            checks.append(self._require_final_message(context, spec))
        if "max_steps" in spec.config:
            configured = True
            checks.append(self._max_steps(context, spec))
        if "min_steps" in spec.config:
            configured = True
            checks.append(self._min_steps(context, spec))
        if "max_tool_calls" in spec.config:
            configured = True
            checks.append(self._max_tool_calls(context, spec))
        if "allowed_statuses" in spec.config:
            configured = True
            checks.append(self._allowed_statuses(context, spec))
        if not configured:
            return [
                criterion(
                    name=spec.name or "trajectory_check",
                    verifier_type=self.verifier_type,
                    score=1.0,
                    weight=spec.weight,
                    passed=True,
                    detail="no trajectory checks configured",
                    required=spec.required,
                )
            ]
        return checks

    def _require_final_message(self, context: VerifierContext, spec: VerifierSpec) -> CriterionResult:
        required = bool(spec.config.get("require_final_message", True))
        final_messages = [step.message or "" for step in context.trajectory.steps if step.kind == "final_message"]
        has_message = any(message.strip() for message in final_messages)
        passed = has_message if required else True
        return criterion(
            name="trajectory_check:final_message",
            verifier_type=self.verifier_type,
            score=1.0 if passed else 0.0,
            weight=float(spec.config.get("final_message_weight", spec.weight)),
            passed=passed,
            detail=f"present={has_message}",
            required=spec.required,
            evidence={"final_message_count": len(final_messages)},
        )

    def _max_steps(self, context: VerifierContext, spec: VerifierSpec) -> CriterionResult:
        limit = int(spec.config["max_steps"])
        count = context.trajectory.final_metrics.n_steps
        passed = count <= limit
        return criterion(
            name="trajectory_check:max_steps",
            verifier_type=self.verifier_type,
            score=1.0 if passed else 0.0,
            weight=float(spec.config.get("max_steps_weight", spec.weight)),
            passed=passed,
            detail=f"steps={count}, limit={limit}",
            required=spec.required,
            evidence={"steps": count, "limit": limit},
        )

    def _min_steps(self, context: VerifierContext, spec: VerifierSpec) -> CriterionResult:
        minimum = int(spec.config["min_steps"])
        count = context.trajectory.final_metrics.n_steps
        passed = count >= minimum
        return criterion(
            name="trajectory_check:min_steps",
            verifier_type=self.verifier_type,
            score=1.0 if passed else 0.0,
            weight=float(spec.config.get("min_steps_weight", spec.weight)),
            passed=passed,
            detail=f"steps={count}, minimum={minimum}",
            required=spec.required,
            evidence={"steps": count, "minimum": minimum},
        )

    def _max_tool_calls(self, context: VerifierContext, spec: VerifierSpec) -> CriterionResult:
        limit = int(spec.config["max_tool_calls"])
        count = context.trajectory.final_metrics.n_tool_calls
        passed = count <= limit
        return criterion(
            name="trajectory_check:max_tool_calls",
            verifier_type=self.verifier_type,
            score=1.0 if passed else 0.0,
            weight=float(spec.config.get("max_tool_calls_weight", spec.weight)),
            passed=passed,
            detail=f"tool_calls={count}, limit={limit}",
            required=spec.required,
            evidence={"tool_calls": count, "limit": limit},
        )

    def _allowed_statuses(self, context: VerifierContext, spec: VerifierSpec) -> CriterionResult:
        allowed = {str(item) for item in spec.config.get("allowed_statuses", [])}
        bad = [step.index for step in context.trajectory.steps if step.status not in allowed]
        passed = not bad
        return criterion(
            name="trajectory_check:allowed_statuses",
            verifier_type=self.verifier_type,
            score=1.0 if passed else 0.0,
            weight=float(spec.config.get("allowed_statuses_weight", spec.weight)),
            passed=passed,
            detail=f"disallowed_step_indexes={bad}",
            required=spec.required,
            evidence={"allowed": sorted(allowed), "bad_step_indexes": bad},
        )
