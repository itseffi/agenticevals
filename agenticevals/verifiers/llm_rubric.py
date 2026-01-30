from __future__ import annotations

import json
from typing import Any

from agenticevals.model_io import anthropic_messages, openai_response
from agenticevals.schema import VerifierSpec
from agenticevals.verifiers.base import BaseVerifier, VerifierContext, criterion
from agenticevals.verifiers.schema import CriterionResult


class LLMRubricVerifier(BaseVerifier):
    verifier_type = "llm_rubric"

    def verify(self, context: VerifierContext, spec: VerifierSpec) -> list[CriterionResult]:
        if "fixture_score" in spec.config or "score" in spec.config:
            score = float(spec.config.get("fixture_score", spec.config.get("score", 0.0)))
            threshold = float(spec.config.get("threshold", 0.5))
            passed = bool(spec.config.get("passed", score >= threshold))
            return [
                criterion(
                    name=spec.name or "llm_rubric",
                    verifier_type=self.verifier_type,
                    score=score,
                    weight=spec.weight,
                    passed=passed,
                    detail=str(spec.config.get("detail", "fixture rubric score")),
                    required=spec.required,
                    deterministic=True,
                    evidence={"threshold": threshold},
                )
            ]
        try:
            judged = self._judge(context, spec)
            score = float(judged.get("score", 0.0))
            threshold = float(spec.config.get("threshold", 0.5))
            passed = bool(judged.get("passed", score >= threshold))
            detail = str(judged.get("reason", ""))
            return [
                criterion(
                    name=spec.name or "llm_rubric",
                    verifier_type=self.verifier_type,
                    score=score,
                    weight=spec.weight,
                    passed=passed,
                    detail=detail,
                    required=spec.required,
                    deterministic=False,
                    evidence={"threshold": threshold, "judge": judged},
                )
            ]
        except Exception as exc:
            return [
                criterion(
                    name=spec.name or "llm_rubric",
                    verifier_type=self.verifier_type,
                    score=0.0,
                    weight=spec.weight,
                    passed=False,
                    detail=f"judge failed: {exc}",
                    required=spec.required,
                    deterministic=False,
                    error=str(exc),
                )
            ]

    def _judge(self, context: VerifierContext, spec: VerifierSpec) -> dict[str, Any]:
        provider = str(spec.config.get("provider", "openai"))
        model = str(spec.config.get("model", "gpt-4o-mini" if provider == "openai" else "claude-sonnet-4-6"))
        timeout = int(spec.config.get("timeout", 60))
        prompt = _rubric_prompt(context, spec)
        if provider == "anthropic":
            response = anthropic_messages(
                model,
                [{"role": "user", "content": prompt}],
                timeout=timeout,
                max_tokens=int(spec.config.get("max_tokens", 1000)),
            )
            return _parse_json_object(response.text)
        if provider == "openai":
            response = openai_response(model, prompt, timeout=timeout)
            return _parse_json_object(response.text)
        raise ValueError(f"unsupported llm_rubric provider: {provider}")


def _rubric_prompt(context: VerifierContext, spec: VerifierSpec) -> str:
    rubric = str(spec.config.get("rubric", "Judge whether the agent satisfied the task."))
    transcript = _compact_transcript(context)
    return (
        "You are grading an AI agent trajectory. Return only strict JSON with keys "
        '"score" (number from 0 to 1), "passed" (boolean), and "reason" (short string).\n\n'
        f"Task:\n{context.task.prompt}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Final response:\n{context.final_response}\n\n"
        f"Trajectory summary:\n{transcript}\n"
    )


def _compact_transcript(context: VerifierContext) -> str:
    rows: list[str] = []
    max_steps = int(context.task.limits.max_steps or 50)
    for step in context.trajectory.steps[:max_steps]:
        if step.message:
            rows.append(f"{step.index}. {step.source}/{step.kind}: {step.message[:500]}")
        for call in step.tool_calls:
            rows.append(f"{step.index}. tool_call {call.name} args={json.dumps(call.arguments, sort_keys=True, default=str)[:500]}")
        for result in step.tool_results:
            status = "error" if result.is_error else "ok"
            rows.append(f"{step.index}. tool_result {result.name} status={status} content={str(result.content)[:500]}")
        if step.observation is not None and not step.tool_results:
            rows.append(f"{step.index}. observation {step.kind}: {str(step.observation)[:500]}")
    return "\n".join(rows)


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)
