from __future__ import annotations

import shutil
from pathlib import Path

from agenticevals.backends.base import Backend, BackendWorkspace
from agenticevals.utils import CommandResult, run_command, safe_relative_path


class LocalBackend(Backend):
    name = "local"

    def create_workspace(self, fixture: Path, run_dir: Path) -> BackendWorkspace:
        workspace = run_dir / "workspace"
        if fixture.is_dir():
            shutil.copytree(fixture, workspace)
        else:
            workspace.mkdir(parents=True)
            shutil.copy2(fixture, workspace / fixture.name)
        backend_workspace = BackendWorkspace(run_dir=run_dir, workspace=workspace)
        run_command("git init", workspace, timeout=30)
        run_command("git add .", workspace, timeout=30)
        run_command(
            "git -c user.email=agenticevals@example.invalid -c user.name=agenticevals commit -m baseline",
            timeout=30,
            cwd=workspace,
        )
        return backend_workspace

    def run(self, workspace: BackendWorkspace, command: str, timeout: int) -> CommandResult:
        return run_command(command, workspace.workspace, timeout=timeout)

    def read_file(self, workspace: BackendWorkspace, path: str) -> str:
        rel = safe_relative_path(path)
        return (workspace.workspace / rel).read_text(encoding="utf-8", errors="replace")

    def write_file(self, workspace: BackendWorkspace, path: str, content: str) -> None:
        rel = safe_relative_path(path)
        target = workspace.workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def changed_files(self, workspace: BackendWorkspace) -> list[str]:
        tracked = self.run(workspace, "git diff --name-only HEAD", timeout=30)
        untracked = self.run(workspace, "git ls-files --others --exclude-standard", timeout=30)
        lines: list[str] = []
        if tracked.ok:
            lines.extend(tracked.stdout.splitlines())
        if untracked.ok:
            lines.extend(untracked.stdout.splitlines())
        return sorted({line.strip() for line in lines if line.strip() and not _generated_path(line.strip())})

    def diff(self, workspace: BackendWorkspace) -> str:
        return self.run(workspace, "git diff -- HEAD", timeout=30).stdout


def _generated_path(path: str) -> bool:
    return path.startswith(".agenticevals/") or "__pycache__/" in path or path.endswith((".pyc", ".pyo"))
