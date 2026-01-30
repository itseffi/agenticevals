from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agenticevals.utils import safe_relative_path


@dataclass(frozen=True)
class FileCheckResult:
    name: str
    passed: bool
    detail: str


def run_file_checks(workspace: Path, checks: list[dict]) -> list[FileCheckResult]:
    results: list[FileCheckResult] = []
    for index, check in enumerate(checks, start=1):
        rel = safe_relative_path(str(check["path"]))
        path = workspace / rel
        name = check.get("name", f"file:{rel}")
        if not path.exists():
            results.append(FileCheckResult(name=name, passed=False, detail=f"{rel} does not exist"))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "contains" in check:
            needle = str(check["contains"])
            passed = needle in text
            results.append(FileCheckResult(name=name, passed=passed, detail=f"contains {needle!r}: {passed}"))
        elif "not_contains" in check:
            needle = str(check["not_contains"])
            passed = needle not in text
            results.append(FileCheckResult(name=name, passed=passed, detail=f"not_contains {needle!r}: {passed}"))
        else:
            results.append(FileCheckResult(name=name, passed=True, detail=f"{rel} exists"))
    return results

