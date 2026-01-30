from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from agenticevals.canonical import normalize_run


def write_viewer(run_dir: Path, output: Path | None = None) -> Path:
    root = run_dir.expanduser().resolve()
    target = output.expanduser().resolve() if output else root / "viewer.html"
    rows = normalize_run(root)
    body = "\n".join(
        "<tr>"
        f"<td>{row.index}</td>"
        f"<td>{html.escape(row.actor)}</td>"
        f"<td>{html.escape(row.action_type)}</td>"
        f"<td>{html.escape(row.status)}</td>"
        f"<td>{html.escape(row.name)}</td>"
        f"<td><pre>{html.escape(json.dumps(row.data, indent=2, sort_keys=True))}</pre></td>"
        "</tr>"
        for row in rows
    )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>agenticevals trajectory viewer</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    pre {{ margin: 0; white-space: pre-wrap; max-width: 720px; }}
    .meta {{ color: #4b5563; }}
  </style>
</head>
<body>
  <h1>Trajectory Viewer</h1>
  <p class="meta"><code>{html.escape(str(root))}</code> · {len(rows)} events</p>
  <table>
    <thead><tr><th>#</th><th>Actor</th><th>Type</th><th>Status</th><th>Name</th><th>Data</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""
    target.write_text(doc, encoding="utf-8")
    return target


def write_review(run_dir: Path, output: Path | None = None) -> Path:
    root = run_dir.expanduser().resolve()
    suite_path = root / "suite.json"
    if not suite_path.exists():
        return write_viewer(root, output=output)
    target = output.expanduser().resolve() if output else root / "review.html"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    result_rows = _read_json(root / "results.json", [])
    failures = _read_json(root / "failures.json", {"clusters": []})
    tasks = suite.get("tasks", [])
    task_rows = "\n".join(_task_row(index, row) for index, row in enumerate(tasks, start=1))
    result_table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('agent', '')))}</td>"
        f"<td>{row.get('tasks', 0)}</td>"
        f"<td>{row.get('passed', 0)}</td>"
        f"<td>{float(row.get('pass_rate', 0.0)):.1%}</td>"
        f"<td>{float(row.get('mean_score', 0.0)):.3f}</td>"
        "</tr>"
        for row in result_rows
    )
    failure_cards = "\n".join(_failure_card(cluster) for cluster in failures.get("clusters", [])) or "<p>No failed tasks in this suite run.</p>"
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>agenticevals suite review</title>
  <style>
    :root {{ color-scheme: light; --border: #d1d5db; --muted: #4b5563; --bg: #f9fafb; --pass: #166534; --fail: #991b1b; }}
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #111827; background: white; }}
    header {{ padding: 24px 32px; border-bottom: 1px solid var(--border); background: var(--bg); }}
    main {{ padding: 24px 32px 40px; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    section {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid var(--border); padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    textarea {{ width: 100%; min-height: 72px; resize: vertical; font: inherit; border: 1px solid var(--border); border-radius: 6px; padding: 8px; }}
    button, select {{ font: inherit; border: 1px solid var(--border); border-radius: 6px; background: white; padding: 6px 10px; }}
    pre {{ margin: 0; white-space: pre-wrap; max-width: 680px; }}
    .meta {{ color: var(--muted); }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 16px; }}
    .metric {{ border: 1px solid var(--border); border-radius: 8px; padding: 12px; background: white; }}
    .metric strong {{ display: block; font-size: 1.4rem; }}
    .pass {{ color: var(--pass); font-weight: 700; }}
    .fail {{ color: var(--fail); font-weight: 700; }}
    .cluster {{ border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin: 10px 0; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <header>
    <h1>Suite Review</h1>
    <p class="meta"><code>{html.escape(str(root))}</code></p>
    <div class="metrics">
      <div class="metric"><span>Total</span><strong>{suite.get('total', 0)}</strong></div>
      <div class="metric"><span>Passed</span><strong>{suite.get('passed', 0)}</strong></div>
      <div class="metric"><span>Pass rate</span><strong>{float(suite.get('pass_rate', 0.0)):.1%}</strong></div>
      <div class="metric"><span>Mean score</span><strong>{float(suite.get('mean_score', 0.0)):.3f}</strong></div>
    </div>
  </header>
  <main>
    <section>
      <h2>Reviewer Notes</h2>
      <textarea id="suite-notes" placeholder="Notes stay in this browser through localStorage."></textarea>
    </section>
    <section>
      <h2>Suite Results</h2>
      <table><thead><tr><th>Agent</th><th>Tasks</th><th>Passed</th><th>Pass rate</th><th>Mean score</th></tr></thead><tbody>{result_table_rows}</tbody></table>
    </section>
    <section>
      <h2>Failure Clusters</h2>
      {failure_cards}
    </section>
    <section>
      <div class="toolbar">
        <h2 style="margin-right:auto">Tasks</h2>
        <label>Status <select id="status-filter"><option value="all">all</option><option value="passed">passed</option><option value="failed">failed</option></select></label>
      </div>
      <table id="tasks"><thead><tr><th>#</th><th>Status</th><th>Task</th><th>Agent</th><th>Trials</th><th>Score</th><th>Run</th><th>Summary</th><th>Review</th></tr></thead><tbody>{task_rows}</tbody></table>
    </section>
  </main>
  <script>
    const rootKey = "agenticevals-review:" + {json.dumps(str(root))};
    const suiteNotes = document.getElementById("suite-notes");
    suiteNotes.value = localStorage.getItem(rootKey + ":notes") || "";
    suiteNotes.addEventListener("input", () => localStorage.setItem(rootKey + ":notes", suiteNotes.value));
    document.querySelectorAll("[data-review]").forEach((el) => {{
      const key = rootKey + ":task:" + el.dataset.review;
      el.value = localStorage.getItem(key) || "";
      el.addEventListener("input", () => localStorage.setItem(key, el.value));
    }});
    document.getElementById("status-filter").addEventListener("change", (event) => {{
      const value = event.target.value;
      document.querySelectorAll("#tasks tbody tr").forEach((row) => {{
        row.classList.toggle("hidden", value !== "all" && row.dataset.status !== value);
      }});
    }});
  </script>
</body>
</html>
"""
    target.write_text(doc, encoding="utf-8")
    return target


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _task_row(index: int, row: dict[str, Any]) -> str:
    passed = bool(row.get("passed"))
    status = "passed" if passed else "failed"
    run_dir = str(row.get("run_dir", ""))
    summary = json.dumps(row.get("summary", {}), indent=2, sort_keys=True)
    return (
        f'<tr data-status="{status}">'
        f"<td>{index}</td>"
        f'<td class="{"pass" if passed else "fail"}">{status.upper()}</td>'
        f"<td>{html.escape(str(row.get('task_id', '')))}</td>"
        f"<td>{html.escape(str(row.get('agent', '')))}</td>"
        f"<td>{row.get('trials', 1)}</td>"
        f"<td>{float(row.get('score', 0.0)):.3f}</td>"
        f"<td><code>{html.escape(run_dir)}</code></td>"
        f"<td><pre>{html.escape(summary)}</pre></td>"
        f'<td><textarea data-review="{html.escape(str(row.get("task_id", index)))}"></textarea></td>'
        "</tr>"
    )


def _failure_card(cluster: dict[str, Any]) -> str:
    items = "\n".join(
        f"<li><code>{html.escape(str(item.get('task_id', '')))}</code> "
        f"{html.escape(str(item.get('agent', '')))} score={float(item.get('score', 0.0)):.3f}</li>"
        for item in cluster.get("items", [])
    )
    return (
        '<div class="cluster">'
        f"<h3>{html.escape(str(cluster.get('label', 'unknown')))} ({cluster.get('count', 0)})</h3>"
        f"<ul>{items}</ul>"
        "</div>"
    )
