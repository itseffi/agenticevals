from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from agenticevals.schema import ServiceSpec
from agenticevals.trace import Trajectory


class ServiceStartError(RuntimeError):
    pass


class ServiceManager:
    def __init__(self, services: list[ServiceSpec], cwd: Path, trace: Trajectory | None = None):
        self.services = services
        self.cwd = cwd
        self.trace = trace
        self._spawned: list[tuple[ServiceSpec, subprocess.Popen]] = []

    def __enter__(self) -> "ServiceManager":
        try:
            for service in self.services:
                if self._is_healthy(service):
                    self._trace("service.ready", service=service.name, reused=True)
                    continue
                self._spawn(service)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for service, proc in reversed(self._spawned):
            self._trace("service.stop", service=service.name, pid=proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._spawned.clear()

    def reset_all(self) -> None:
        for service in self.services:
            if service.reset_endpoint:
                self._request_json(service.reset_endpoint, method="POST", timeout=5)
                self._trace("service.reset", service=service.name)

    def audit_all(self) -> dict[str, Any]:
        audit: dict[str, Any] = {}
        for service in self.services:
            endpoint = service.audit_endpoint or _audit_endpoint_from_reset(service.reset_endpoint)
            if not endpoint:
                continue
            try:
                audit[service.name] = self._request_json(endpoint, method="GET", timeout=5)
            except Exception as exc:
                audit[service.name] = {"error": str(exc)}
            self._trace("service.audit", service=service.name, audit=audit[service.name])
        return audit

    def _spawn(self, service: ServiceSpec) -> None:
        cmd = shlex.split(service.command)
        if cmd and cmd[0] in {"python", "python3"}:
            cmd[0] = sys.executable
        env = dict(os.environ)
        env.update(service.env)
        env["PORT"] = str(service.port)
        self._trace("service.start", service=service.name, command=service.command, port=service.port)
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + service.ready_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise ServiceStartError(f"service {service.name} exited with {proc.returncode}: {stderr[-1000:]}")
            if self._is_healthy(service):
                self._spawned.append((service, proc))
                self._trace("service.ready", service=service.name, reused=False, pid=proc.pid)
                return
            time.sleep(0.2)
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        raise ServiceStartError(f"service {service.name} did not become ready within {service.ready_timeout}s")

    def _is_healthy(self, service: ServiceSpec) -> bool:
        try:
            self._request_json(service.health_check, method=service.health_check_method, timeout=2)
            return True
        except Exception:
            return False

    def _request_json(self, url: str, method: str, timeout: int) -> Any:
        return _request_json(url, method=method, timeout=timeout)

    def _trace(self, event_type: str, **data: Any) -> None:
        if self.trace is not None:
            self.trace.add(event_type, **data)


def _request_json(url: str, method: str, timeout: int) -> Any:
    request = urllib.request.Request(url, data=b"{}" if method.upper() == "POST" else None, headers={"Content-Type": "application/json"}, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": raw, "status": exc.code}


def _audit_endpoint_from_reset(reset_endpoint: str | None) -> str | None:
    if not reset_endpoint:
        return None
    if reset_endpoint.endswith("/reset"):
        return reset_endpoint[: -len("/reset")] + "/audit"
    return None
