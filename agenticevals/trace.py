from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    type: str
    data: dict[str, Any]
    ts: float = field(default_factory=time.time)


@dataclass
class Trajectory:
    task_id: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[TraceEvent] = field(default_factory=list)

    def add(self, event_type: str, **data: Any) -> None:
        self.events.append(TraceEvent(type=event_type, data=data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "events": [asdict(event) for event in self.events],
        }

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

