from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agenticevals.utils import CommandResult


@dataclass(frozen=True)
class BackendWorkspace:
    run_dir: Path
    workspace: Path


class Backend:
    name = "backend"

    def create_workspace(self, fixture: Path, run_dir: Path) -> BackendWorkspace:
        raise NotImplementedError

    def run(self, workspace: BackendWorkspace, command: str, timeout: int) -> CommandResult:
        raise NotImplementedError

    def read_file(self, workspace: BackendWorkspace, path: str) -> str:
        raise NotImplementedError

    def write_file(self, workspace: BackendWorkspace, path: str, content: str) -> None:
        raise NotImplementedError

    def changed_files(self, workspace: BackendWorkspace) -> list[str]:
        raise NotImplementedError

    def diff(self, workspace: BackendWorkspace) -> str:
        raise NotImplementedError

    def cleanup(self, workspace: BackendWorkspace) -> None:
        pass

