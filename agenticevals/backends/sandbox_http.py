from __future__ import annotations

from pathlib import Path

from agenticevals.backends.base import BackendWorkspace
from agenticevals.backends.local import LocalBackend
from agenticevals.sandbox import SandboxClient, SandboxServer
from agenticevals.trace import Trajectory
from agenticevals.utils import CommandResult


class SandboxHTTPBackend(LocalBackend):
    name = "sandbox-http"

    def __init__(self):
        self._servers: dict[Path, SandboxServer] = {}
        self._clients: dict[Path, SandboxClient] = {}

    def create_workspace(self, fixture: Path, run_dir: Path) -> BackendWorkspace:
        workspace = super().create_workspace(fixture, run_dir)
        trace = Trajectory(task_id=f"sandbox-http-{run_dir.name}")
        server = SandboxServer(workspace.workspace, trace).__enter__()
        self._servers[workspace.workspace] = server
        self._clients[workspace.workspace] = server.client
        return workspace

    def run(self, workspace: BackendWorkspace, command: str, timeout: int) -> CommandResult:
        payload = self._clients[workspace.workspace].exec(command, timeout_seconds=timeout)
        return CommandResult(
            command=command,
            returncode=int(payload.get("exit_code", 1)),
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
            timed_out=payload.get("exit_code") == -1,
        )

    def read_file(self, workspace: BackendWorkspace, path: str) -> str:
        return str(self._clients[workspace.workspace].read(path).get("content", ""))

    def write_file(self, workspace: BackendWorkspace, path: str, content: str) -> None:
        self._clients[workspace.workspace].write(path, content)

    def cleanup(self, workspace: BackendWorkspace) -> None:
        server = self._servers.pop(workspace.workspace, None)
        self._clients.pop(workspace.workspace, None)
        if server is not None:
            server.__exit__(None, None, None)
