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
                    detail=str(spec.config.get("detail", "fixture rubric score (not a real LLM judgment)")),
                    required=spec.required,
                    deterministic=True,
                    evidence={"threshold": threshold, "fixture": True},
                )
            ]
        threshold = float(spec.config.get("threshold", 0.5))
        try:
            judged = self._judge(context, spec)
        except Exception as exc:
            # The judge erroring (transient API failure, unparseable output) is
            # not evidence about the agent. Abstain: weight 0 so it cannot drag
            # down the weighted reward, and non-required so it cannot fail an
            # otherwise-passing run. The error stays recorded for inspection.
            return [
                criterion(
                    name=spec.name or "llm_rubric",
                    verifier_type=self.verifier_type,
                    score=0.0,
                    weight=0.0,
                    passed=False,
                    detail=f"judge abstained: {exc}",
                    required=False,
                    deterministic=False,
                    error=str(exc),
                )
            ]
        if "passed" in judged:
            passed = bool(judged["passed"])
            score = float(judged.get("score", 1.0 if passed else 0.0))
        else:
            score = float(judged.get("score", 0.0))
            passed = score >= threshold
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
    transcript = _compact_transcript(context, spec)
    return (
        "You are grading an AI agent trajectory against a rubric. First reason about "
        "whether the agent satisfied the task, then give a single binary verdict. "
        "Return only strict JSON with keys "
        '"reason" (short string justifying the verdict, written first) and '
        '"passed" (boolean: true only if the rubric is fully satisfied). '
        'You may optionally include "score" (number from 0 to 1).\n\n'
        f"Task:\n{context.task.prompt}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Final response:\n{context.final_response}\n\n"
        f"Trajectory summary:\n{transcript}\n"
    )


def _compact_transcript(context: VerifierContext, spec: VerifierSpec) -> str:
    rows: list[str] = []
    max_steps = int(spec.config.get("transcript_max_steps", context.task.limits.max_steps or 50))
    # Per-field character cap. Truncating too aggressively can hide the actual
    # failure from the judge, so make it tunable per rubric.
    cap = int(spec.config.get("transcript_max_chars", 500))
    for step in context.trajectory.steps[:max_steps]:
        if step.message:
            rows.append(f"{step.index}. {step.source}/{step.kind}: {step.message[:cap]}")
        for call in step.tool_calls:
            rows.append(f"{step.index}. tool_call {call.name} args={json.dumps(call.arguments, sort_keys=True, default=str)[:cap]}")
        for result in step.tool_results:
            status = "error" if result.is_error else "ok"
            rows.append(f"{step.index}. tool_result {result.name} status={status} content={str(result.content)[:cap]}")
        if step.observation is not None and not step.tool_results:
            rows.append(f"{step.index}. observation {step.kind}: {str(step.observation)[:cap]}")
    return "\n".join(rows)


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Models often wrap JSON in prose or code fences. Decode the first balanced
    # object rather than failing (which would force the judge to abstain).
    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in judge response")
