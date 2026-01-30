from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from agenticevals.scorers import EvalScore, ScoreItem


REWARD_SCHEMA_VERSION = "agenticevals.reward.v1"
REWARD_DETAILS_SCHEMA_VERSION = "agenticevals.reward-details.v1"


@dataclass(frozen=True)
class CriterionResult:
    name: str
    verifier_type: str
    score: float
    weight: float
    passed: bool
    detail: str = ""
    required: bool = True
    deterministic: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def value(self) -> float:
        return self.score * self.weight

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifierRunResult:
    criteria: list[CriterionResult]

    @property
    def reward(self) -> float:
        total = sum(item.weight for item in self.criteria)
        if total <= 0:
            return 1.0
        return sum(item.value for item in self.criteria) / total

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.criteria if item.required)

    def to_reward_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REWARD_SCHEMA_VERSION,
            "reward": round(self.reward, 6),
            "max_reward": 1.0,
            "passed": self.passed,
            "rewards": _reward_map(self.criteria),
            "weights": {item.name: item.weight for item in self.criteria},
            "components": [
                {
                    "name": item.name,
                    "type": item.verifier_type,
                    "score": round(item.score, 6),
                    "value": round(item.value, 6),
                    "max_value": item.weight,
                    "passed": item.passed,
                    "required": item.required,
                    "detail": item.detail,
                }
                for item in self.criteria
            ],
        }

    def to_details_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REWARD_DETAILS_SCHEMA_VERSION,
            "passed": self.passed,
            "reward": round(self.reward, 6),
            "criteria": [item.to_dict() for item in self.criteria],
        }

    def to_score(self) -> EvalScore:
        items = [
            ScoreItem(
                name=item.name,
                passed=item.passed,
                points=item.value,
                max_points=item.weight,
                detail=item.detail or json.dumps(item.evidence, sort_keys=True, default=str),
            )
            for item in self.criteria
        ]
        points = sum(item.points for item in items)
        max_points = sum(item.max_points for item in items)
        return EvalScore(passed=self.passed, points=points, max_points=max_points, items=items)


def score_to_verifier_result(score: EvalScore) -> VerifierRunResult:
    criteria = []
    for item in score.items:
        criteria.append(
            CriterionResult(
                name=item.name,
                verifier_type="compat_score",
                score=item.points / item.max_points if item.max_points else 1.0,
                weight=item.max_points,
                passed=item.passed,
                detail=item.detail,
                evidence={"points": item.points, "max_points": item.max_points},
            )
        )
    return VerifierRunResult(criteria=criteria)


def _reward_map(criteria: list[CriterionResult]) -> dict[str, float]:
    rewards: dict[str, float] = {}
    for item in criteria:
        rewards[item.name] = round(item.score, 6)
    return rewards
