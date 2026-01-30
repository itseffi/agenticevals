from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticevals.schema import TaskSpec
from agenticevals.trace import Trajectory


@dataclass(frozen=True)
class AgentRun:
    ok: bool
    final_message: str
    metadata: dict


class BaseAgent:
    name = "base"

    def run(self, task: TaskSpec, workspace: Path, trace: Trajectory, timeout: int, computer: Any = None) -> AgentRun:
        raise NotImplementedError


class AdapterUnavailable(RuntimeError):
    pass
