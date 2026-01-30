from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from agenticevals.backends.base import Backend, BackendWorkspace
from agenticevals.computer.browser import BrowserSession
from agenticevals.trace import Trajectory
from agenticevals.utils import CommandResult, safe_relative_path


@dataclass(frozen=True)
class ComputerContext:
    backend: Backend
    backend_workspace: BackendWorkspace
    trace: Trajectory
    default_timeout: int = 60

    @property
    def workspace(self) -> Path:
        return self.backend_workspace.workspace

    @property
    def run_dir(self) -> Path:
        return self.backend_workspace.run_dir

    def terminal(self, command: str, timeout: int | None = None) -> CommandResult:
        self.trace.add("computer.terminal", command=command, cwd=str(self.workspace), timeout=timeout or self.default_timeout)
        result = self.backend.run(self.backend_workspace, command, timeout=timeout or self.default_timeout)
        self.trace.add(
            "computer.terminal.result",
            command=command,
            returncode=result.returncode,
            stdout=result.stdout[-4000:],
            stderr=result.stderr[-4000:],
            timed_out=result.timed_out,
        )
        return result

    def read_file(self, path: str) -> str:
        rel = safe_relative_path(path)
        text = self.backend.read_file(self.backend_workspace, str(rel))
        self.trace.add("computer.file.read", path=str(rel), bytes=len(text.encode("utf-8")))
        return text

    def write_file(self, path: str, content: str) -> None:
        rel = safe_relative_path(path)
        self.backend.write_file(self.backend_workspace, str(rel), content)
        self.trace.add("computer.file.write", path=str(rel), bytes=len(content.encode("utf-8")))

    def copy_artifact(self, path: str, name: str | None = None) -> Path:
        rel = safe_relative_path(path)
        source = self.workspace / rel
        if not source.exists():
            raise FileNotFoundError(source)
        artifact_dir = self.run_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        target = artifact_dir / (name or rel.name)
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        self.trace.add("computer.artifact.copy", source=str(rel), target=str(target))
        return target

    def changed_files(self) -> list[str]:
        files = self.backend.changed_files(self.backend_workspace)
        self.trace.add("computer.git.changed_files", files=files)
        return files

    def diff(self) -> str:
        diff = self.backend.diff(self.backend_workspace)
        self.trace.add("computer.git.diff", bytes=len(diff.encode("utf-8")))
        return diff

    def browser(self, base_url: str | None = None, timeout: int | None = None) -> BrowserSession:
        artifact_dir = self.run_dir / "artifacts" / "browser"
        self.trace.add("computer.browser.start", base_url=base_url, timeout=timeout or self.default_timeout)
        return BrowserSession(base_url=base_url, timeout=timeout or self.default_timeout, artifact_dir=artifact_dir)
