from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agenticevals.sandbox.client import SandboxClient
from agenticevals.trace import Trajectory


@dataclass
class SandboxServer:
    workspace: Path
    trace: Trajectory
    port: int | None = None

    def __post_init__(self) -> None:
        self.port = self.port or _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.client = SandboxClient(self.base_url)
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "SandboxServer":
        env = dict(os.environ)
        env["PORT"] = str(self.port)
        env["AGENTICEVALS_SANDBOX_WORKSPACE"] = str(self.workspace)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "agenticevals.sandbox.server"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.trace.add("sandbox.start", url=self.base_url, workspace=str(self.workspace), pid=self.process.pid)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"sandbox server exited early with code {self.process.returncode}")
            try:
                health = self.client.get("/health")
                if health.get("status") == "ok":
                    self.trace.add("sandbox.ready", url=self.base_url)
                    return self
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("sandbox server did not become healthy")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process is None:
            return
        self.trace.add("sandbox.stop", url=self.base_url, pid=self.process.pid)
        self.process.terminate()
        try:
            stdout, stderr = self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            stdout, stderr = self.process.communicate(timeout=5)
        self.trace.add(
            "sandbox.output",
            url=self.base_url,
            returncode=self.process.returncode,
            stdout=(stdout or "")[-4000:],
            stderr=(stderr or "")[-4000:],
        )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
