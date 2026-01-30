from __future__ import annotations

import html
import json
from pathlib import Path

from agenticevals.schema import TaskSpec
from agenticevals.scorers import EvalScore
from agenticevals.trace import Trajectory


def write_json_report(path: Path, task: TaskSpec, score: EvalScore, trajectory: Trajectory, changed_files: list[str]) -> None:
    payload = {
        "task": {"id": task.id, "title": task.title},
        "run_id": trajectory.run_id,
        "score": score.to_dict(),
        "changed_files": changed_files,
        "event_counts": _event_counts(trajectory),
        "artifacts": {
            "trajectory_jsonl": "trajectory.jsonl",
            "trajectory_json": "trajectory.json",
            "reward": "reward.json",
            "reward_details": "reward-details.json",
            "score": "score.json",
            "diff": "diff.patch",
            "html_report": "report.html",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_html_report(path: Path, task: TaskSpec, score: EvalScore, trajectory: Trajectory, changed_files: list[str]) -> None:
    rows = "\n".join(
        f"<tr><td>{html.escape(item.name)}</td><td>{'PASS' if item.passed else 'FAIL'}</td>"
        f"<td>{item.points:.1f}/{item.max_points:.1f}</td><td>{html.escape(item.detail)}</td></tr>"
        for item in score.items
    )
    events = html.escape(json.dumps(trajectory.to_dict(), indent=2))
    changed = html.escape("\n".join(changed_files) or "(none)")
    status = "PASS" if score.passed else "FAIL"
    color = "#116329" if score.passed else "#b42318"
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>agenticevals report: {html.escape(task.id)}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .status {{ color: {color}; font-weight: 800; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    pre {{ background: #111827; color: #f9fafb; padding: 16px; overflow: auto; border-radius: 8px; }}
    .meta {{ color: #4b5563; }}
  </style>
</head>
<body>
  <h1>{html.escape(task.title)}</h1>
  <p class="meta">Task <code>{html.escape(task.id)}</code> · Run <code>{html.escape(trajectory.run_id)}</code></p>
  <h2 class="status">{status} · {score.points:.1f}/{score.max_points:.1f}</h2>
  <h2>Score Items</h2>
  <table>
    <thead><tr><th>Name</th><th>Status</th><th>Points</th><th>Detail</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <h2>Changed Files</h2>
  <pre>{changed}</pre>
  <h2>Trajectory</h2>
  <pre>{events}</pre>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _event_counts(trajectory: Trajectory) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in trajectory.events:
        counts[event.type] = counts.get(event.type, 0) + 1
    return dict(sorted(counts.items()))
