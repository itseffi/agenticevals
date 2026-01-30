from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticevals.trace import Trajectory


def ingest_jsonl_trace(path: Path, trace: Trajectory, *, source: str) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                trace.add("agent.trace_ingest.error", source=source, line=line_no, error=str(exc))
                continue
            trace.add("agent.trace", source=source, event=_normalize_event(event))
            count += 1
    trace.add("agent.trace_ingest.finish", source=source, path=str(path), events=count)
    return count


def _normalize_event(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    return {"value": event}
