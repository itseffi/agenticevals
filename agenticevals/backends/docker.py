from __future__ import annotations

import hashlib
import shlex
import shutil
from pathlib import Path

from agenticevals.backends.base import BackendWorkspace
from agenticevals.backends.local import LocalBackend
from agenticevals.utils import CommandResult, run_command


class DockerBackend(LocalBackend):
    name = "docker"

    def __init__(self, image: str = "python:3.12"):
        self.image = image

    def run(self, workspace: BackendWorkspace, command: str, timeout: int) -> CommandResult:
        if shutil.which("docker") is None:
            return CommandResult(command=command, returncode=127, stdout="", stderr="docker executable not found")
        image = self._image_for_workspace(workspace, timeout=timeout)
        if isinstance(image, CommandResult):
            return image
        docker_command = (
            "docker run --rm "
            f"-v {shlex.quote(str(workspace.workspace))}:/workspace "
            "-w /workspace "
            f"{shlex.quote(image)} "
            f"sh -lc {shlex.quote(command)}"
        )
        return run_command(docker_command, workspace.workspace, timeout=timeout)

    def _image_for_workspace(self, workspace: BackendWorkspace, timeout: int) -> str | CommandResult:
        if self.image not in {"auto", "dockerfile"}:
            return self.image
        dockerfile = workspace.workspace / "Dockerfile"
        if not dockerfile.exists():
            return "python:3.12"
        tag = _cached_image_tag(workspace.workspace, dockerfile)
        inspect = run_command(f"docker image inspect {shlex.quote(tag)}", workspace.workspace, timeout=30)
        if inspect.ok:
            return tag
        build = run_command(f"docker build -t {shlex.quote(tag)} -f {shlex.quote(str(dockerfile))} .", workspace.workspace, timeout=timeout)
        if not build.ok:
            return build
        return tag


def _cached_image_tag(workspace: Path, dockerfile: Path) -> str:
    digest = hashlib.sha256()
    digest.update(dockerfile.read_bytes())
    for path in sorted(workspace.glob("**/*")):
        if path.is_file() and ".git" not in path.parts:
            rel = path.relative_to(workspace)
            if rel.parts and rel.parts[0] in {".agenticevals", "__pycache__"}:
                continue
            digest.update(str(rel).encode("utf-8"))
            digest.update(path.read_bytes())
    return f"agenticevals:{digest.hexdigest()[:16]}"
