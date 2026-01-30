from __future__ import annotations

import json
import os
from pathlib import Path


workspace = Path(os.environ["AGENTICEVALS_WORKSPACE"])
trace_path = Path(os.environ["AGENTICEVALS_TRACE_PATH"])
result_path = Path(os.environ["AGENTICEVALS_RESULT_PATH"])
target = workspace / "src" / "text_tools.py"

trace_path.parent.mkdir(parents=True, exist_ok=True)
trace_path.write_text(json.dumps({"type": "edit", "path": "src/text_tools.py"}) + "\n", encoding="utf-8")
target.write_text(
    'import re\n\n\n'
    'def slugify(text: str) -> str:\n'
    '    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower())\n'
    '    return normalized.strip("-")\n',
    encoding="utf-8",
)
result_path.write_text(
    json.dumps({"ok": True, "final_message": "Updated slugify from an external command agent."}),
    encoding="utf-8",
)
