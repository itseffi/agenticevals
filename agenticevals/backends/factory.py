from __future__ import annotations

from agenticevals.backends.base import Backend
from agenticevals.backends.docker import DockerBackend
from agenticevals.backends.local import LocalBackend
from agenticevals.backends.sandbox_http import SandboxHTTPBackend


def create_backend(name: str = "local", image: str | None = None) -> Backend:
    if name == "local":
        return LocalBackend()
    if name == "docker":
        return DockerBackend(image=image or "python:3.12")
    if name == "sandbox-http":
        return SandboxHTTPBackend()
    raise ValueError(f"Unknown backend: {name}")
