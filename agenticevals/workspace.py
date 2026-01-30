from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .schema import TaskSpec
from .utils import CommandResult, run_command


@dataclass(frozen=True)
class Workspace:
    run_dir: Path
    path: Path


class WorkspaceManager:
    def __init__(self, runs_path: Path):
        self.runs_path = runs_path

    def create(self, task: TaskSpec) -> Workspace:
        fixture = task.resolve_fixture()
        if not fixture.exists():
            raise FileNotFoundError(f"Fixture path does not exist: {fixture}")
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir = self.runs_path / f"{task.id}-{stamp}"
        workspace = run_dir / "workspace"
        run_dir.mkdir(parents=True, exist_ok=False)
        if fixture.is_dir():
            shutil.copytree(fixture, workspace)
        else:
            workspace.mkdir()
            shutil.copy2(fixture, workspace / fixture.name)
        self._init_git(workspace)
        return Workspace(run_dir=run_dir, path=workspace)

    def _init_git(self, workspace: Path) -> None:
        if (workspace / ".git").exists():
            return
        run_command("git init", workspace, timeout=30)
        run_command("git add .", workspace, timeout=30)
        run_command(
            "git -c user.email=agenticevals@example.invalid -c user.name=agenticevals commit -m baseline",
            workspace,
            timeout=30,
        )

    @staticmethod
    def changed_files(workspace: Path) -> list[str]:
        tracked = run_command("git diff --name-only HEAD", workspace, timeout=30)
        untracked = run_command("git ls-files --others --exclude-standard", workspace, timeout=30)
        lines: list[str] = []
        if tracked.ok:
            lines.extend(tracked.stdout.splitlines())
        if untracked.ok:
            lines.extend(untracked.stdout.splitlines())
        return sorted({line.strip() for line in lines if line.strip() and not _generated_path(line.strip())})

    @staticmethod
    def diff(workspace: Path) -> CommandResult:
        return run_command("git diff -- HEAD", workspace, timeout=30)


def _generated_path(path: str) -> bool:
    return path.startswith(".agenticevals/") or "__pycache__/" in path or path.endswith((".pyc", ".pyo"))
