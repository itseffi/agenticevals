from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def filtered_review_rows(run_dir: Path, filters: list[str], *, limit: int = 20) -> list[dict[str, Any]]:
    root = run_dir.expanduser().resolve()
    suite = _read_json(root / "suite.json", {})
    rows = list(suite.get("tasks", []))
    parsed = [_parse_filter(item) for item in filters]
    matched = []
    for row in rows:
        if all(str(_field(row, key)) == value for key, value in parsed):
            matched.append(_summary_row(row))
        if len(matched) >= limit:
            break
    return matched


def format_review_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no matching runs"
    lines = []
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        lines.append(f"{status}\t{row['agent']}\t{row['task_id']}\t{row['score']:.3f}\t{row['run_dir']}")
        if row["failed_items"]:
            lines.append("  failed: " + ", ".join(row["failed_items"][:8]))
    return "\n".join(lines)


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    summary = row.get("summary", {})
    failed = [str(item.get("name")) for item in summary.get("items", []) if not item.get("passed", True)]
    return {
        "task_id": row.get("task_id", ""),
        "agent": row.get("agent", ""),
        "passed": bool(row.get("passed", False)),
        "status": "passed" if row.get("passed", False) else "failed",
        "score": float(row.get("score", 0.0) or 0.0),
        "run_dir": row.get("run_dir", ""),
        "failed_items": failed,
    }


def _parse_filter(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"filter must be key=value: {raw}")
    key, value = raw.split("=", 1)
    return key.strip(), value.strip()


def _field(row: dict[str, Any], key: str) -> Any:
    if key == "status":
        return "passed" if row.get("passed", False) else "failed"
    if key in row:
        return row[key]
    return _summary_row(row).get(key)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
