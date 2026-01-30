from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agenticevals.rewards import Reward


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        return cls(
            role=str(data["role"]),
            content=str(data["content"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class AgentResult:
    ok: bool
    final_response: str
    turns: list[Turn] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "final_response": self.final_response,
            "turns": [turn.to_dict() for turn in self.turns],
            "actions": self.actions,
            "observations": self.observations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResult":
        return cls(
            ok=bool(data["ok"]),
            final_response=str(data["final_response"]),
            turns=[Turn.from_dict(item) for item in data.get("turns", [])],
            actions=list(data.get("actions", [])),
            observations=list(data.get("observations", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class RolloutResult:
    run_id: str
    environment: str
    item_id: str
    agent: str
    status: str
    workspace: Path
    run_dir: Path
    started_at: float
    completed_at: float
    prompt: str
    agent_result: AgentResult
    reward: Reward
    artifacts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def new_id(cls) -> str:
        return uuid.uuid4().hex[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "environment": self.environment,
            "item_id": self.item_id,
            "agent": self.agent,
            "status": self.status,
            "workspace": str(self.workspace),
            "run_dir": str(self.run_dir),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.completed_at - self.started_at,
            "prompt": self.prompt,
            "agent_result": self.agent_result.to_dict(),
            "reward": self.reward.to_dict(),
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutResult":
        return cls(
            run_id=str(data["run_id"]),
            environment=str(data["environment"]),
            item_id=str(data["item_id"]),
            agent=str(data["agent"]),
            status=str(data["status"]),
            workspace=Path(data["workspace"]),
            run_dir=Path(data["run_dir"]),
            started_at=float(data["started_at"]),
            completed_at=float(data["completed_at"]),
            prompt=str(data["prompt"]),
            agent_result=AgentResult.from_dict(data["agent_result"]),
            reward=Reward.from_dict(data["reward"]),
            artifacts=dict(data.get("artifacts", {})),
        )


@dataclass(frozen=True)
class EvalResult:
    environment: str
    agent: str
    run_dir: Path
    started_at: float
    completed_at: float
    rollouts: list[RolloutResult]

    def to_dict(self) -> dict[str, Any]:
        total = len(self.rollouts)
        passed = sum(1 for rollout in self.rollouts if rollout.reward.passed)
        reward_sum = sum(rollout.reward.value for rollout in self.rollouts)
        max_sum = sum(rollout.reward.max_value for rollout in self.rollouts)
        return {
            "environment": self.environment,
            "agent": self.agent,
            "run_dir": str(self.run_dir),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.completed_at - self.started_at,
            "total": total,
            "passed": passed,
            "pass_rate": passed / total if total else 0.0,
            "reward": reward_sum,
            "max_reward": max_sum,
            "reward_rate": reward_sum / max_sum if max_sum else 0.0,
            "mean_duration_seconds": (
                sum(rollout.completed_at - rollout.started_at for rollout in self.rollouts) / total if total else 0.0
            ),
            "status_counts": {
                status: sum(1 for rollout in self.rollouts if rollout.status == status)
                for status in sorted({rollout.status for rollout in self.rollouts})
            },
            "reward_components": _component_averages(self.rollouts),
            "rollouts": [rollout.to_dict() for rollout in self.rollouts],
        }


def now() -> float:
    return time.time()


def _component_averages(rollouts: list[RolloutResult]) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for rollout in rollouts:
        for component in rollout.reward.components:
            slot = totals.setdefault(component.name, {"value": 0.0, "max_value": 0.0, "passed": 0.0, "count": 0.0})
            slot["value"] += component.value
            slot["max_value"] += component.max_value
            slot["passed"] += 1.0 if component.passed else 0.0
            slot["count"] += 1.0
    return {
        name: {
            "mean_value": values["value"] / values["count"] if values["count"] else 0.0,
            "mean_max_value": values["max_value"] / values["count"] if values["count"] else 0.0,
            "pass_rate": values["passed"] / values["count"] if values["count"] else 0.0,
        }
        for name, values in totals.items()
    }
