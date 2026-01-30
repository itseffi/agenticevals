from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RewardComponent:
    name: str
    value: float
    max_value: float
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardComponent":
        return cls(
            name=str(data["name"]),
            value=float(data["value"]),
            max_value=float(data["max_value"]),
            passed=bool(data["passed"]),
            detail=str(data.get("detail", "")),
        )


@dataclass(frozen=True)
class Reward:
    value: float
    max_value: float
    passed: bool
    components: list[RewardComponent] = field(default_factory=list)

    @classmethod
    def from_components(cls, components: list[RewardComponent]) -> "Reward":
        value = sum(component.value for component in components)
        max_value = sum(component.max_value for component in components)
        return cls(
            value=value,
            max_value=max_value,
            passed=all(component.passed for component in components),
            components=components,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "max_value": self.max_value,
            "passed": self.passed,
            "components": [component.to_dict() for component in self.components],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reward":
        return cls(
            value=float(data["value"]),
            max_value=float(data["max_value"]),
            passed=bool(data["passed"]),
            components=[RewardComponent.from_dict(item) for item in data.get("components", [])],
        )
