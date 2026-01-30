from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SandboxClient:
    base_url: str
    timeout: int = 30

    def exec(self, command: str, timeout_seconds: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"command": command}
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        return self.post("/exec", payload)

    def read(self, path: str) -> dict[str, Any]:
        return self.post("/read", {"path": path})

    def write(self, path: str, content: str) -> dict[str, Any]:
        return self.post("/write", {"path": path, "content": content})

    def browser_goto(self, url: str, save_as: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"url": url}
        if save_as:
            payload["save_as"] = save_as
        return self.post("/browser/goto", payload)

    def browser_check(self, url: str, contains: str = "") -> dict[str, Any]:
        return self.post("/browser/check", {"url": url, "contains": contains})

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def get(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(self.base_url.rstrip("/") + path, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
