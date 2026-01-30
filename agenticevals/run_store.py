from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticevals.rollouts import RolloutResult


@dataclass(frozen=True)
class EvalRunStore:
    path: Path

    @classmethod
    def create(cls, runs_path: Path, environment_name: str) -> "EvalRunStore":
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = uuid.uuid4().hex[:8]
        path = runs_path / f"eval-{environment_name}-{stamp}-{run_id}"
        path.mkdir(parents=True, exist_ok=False)
        (path / "rollouts").mkdir()
        return cls(path=path)

    @classmethod
    def resume(cls, path: Path) -> "EvalRunStore":
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Eval run does not exist: {resolved}")
        (resolved / "rollouts").mkdir(exist_ok=True)
        return cls(path=resolved)

    @property
    def rollouts_dir(self) -> Path:
        return self.path / "rollouts"

    def rollout_dir(self, item_id: str, run_id: str) -> Path:
        safe_item = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in item_id)
        path = self.rollouts_dir / f"{safe_item}-{run_id}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def completed_item_ids(self) -> set[str]:
        completed: set[str] = set()
        for path in self.rollouts_dir.glob("*/rollout.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if data.get("item_id") and data.get("status") in {"passed", "failed", "partial", "error", "timeout"}:
                completed.add(str(data["item_id"]))
        return completed

    def load_rollouts(self) -> list[RolloutResult]:
        rollouts: list[RolloutResult] = []
        for path in sorted(self.rollouts_dir.glob("*/rollout.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rollouts.append(RolloutResult.from_dict(data))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return rollouts

    def write_eval(self, data: dict[str, Any]) -> Path:
        target = self.path / "eval.json"
        target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return target
