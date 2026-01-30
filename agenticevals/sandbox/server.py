from __future__ import annotations

import base64
import glob
import json
import os
import subprocess
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


WORKSPACE = Path(os.environ.get("AGENTICEVALS_SANDBOX_WORKSPACE", "/workspace")).resolve()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"status": "ok", "workspace": str(WORKSPACE)})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        body = self._read_json()
        if self.path == "/exec":
            self._exec(body)
        elif self.path == "/read":
            self._read(body)
        elif self.path == "/write":
            self._write(body)
        elif self.path == "/edit":
            self._edit(body)
        elif self.path == "/glob":
            self._glob(body)
        elif self.path == "/grep":
            self._grep(body)
        elif self.path == "/download":
            self._download(body)
        elif self.path == "/browser/goto":
            self._browser_goto(body)
        elif self.path == "/browser/check":
            self._browser_check(body)
        else:
            self._json({"error": "not found"}, status=404)

    def _exec(self, body: dict[str, Any]) -> None:
        try:
            proc = subprocess.run(
                str(body["command"]),
                cwd=str(WORKSPACE),
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=int(body.get("timeout_seconds", 30)),
            )
            self._json({"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
        except subprocess.TimeoutExpired:
            self._json({"exit_code": -1, "stdout": "", "stderr": "timed out"})

    def _read(self, body: dict[str, Any]) -> None:
        path = _safe_path(str(body.get("path") or body.get("file_path") or ""))
        if not path.exists():
            self._json({"error": f"file not found: {path}"}, status=404)
            return
        self._json({"content": path.read_text(encoding="utf-8", errors="replace"), "encoding": "utf-8"})

    def _write(self, body: dict[str, Any]) -> None:
        path = _safe_path(str(body.get("path") or body.get("file_path") or ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(body.get("content", ""))
        path.write_text(content, encoding="utf-8")
        self._json({"written": str(path), "bytes": len(content.encode("utf-8"))})

    def _edit(self, body: dict[str, Any]) -> None:
        path = _safe_path(str(body.get("path") or body.get("file_path") or ""))
        content = path.read_text(encoding="utf-8", errors="replace")
        old = str(body["old_string"])
        new = str(body["new_string"])
        count = content.count(old)
        if count == 0:
            self._json({"error": "old_string not found"}, status=400)
            return
        if count > 1 and not body.get("replace_all", False):
            self._json({"error": "old_string not unique"}, status=400)
            return
        path.write_text(content.replace(old, new, -1 if body.get("replace_all", False) else 1), encoding="utf-8")
        self._json({"edited": str(path), "replacements": count if body.get("replace_all", False) else 1})

    def _glob(self, body: dict[str, Any]) -> None:
        base = _safe_path(str(body.get("path") or "."))
        pattern = str(base / str(body["pattern"]))
        files = [{"path": item, "size_bytes": Path(item).stat().st_size} for item in sorted(glob.glob(pattern, recursive=True)) if Path(item).is_file()]
        self._json({"files": files[: int(body.get("max_files", 50))]})

    def _grep(self, body: dict[str, Any]) -> None:
        command = ["grep", "-R", "-n", str(body["pattern"]), str(_safe_path(str(body.get("path") or ".")))]
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self._json({"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})

    def _download(self, body: dict[str, Any]) -> None:
        path = _safe_path(str(body["path"]))
        raw = path.read_bytes()
        self._json({"path": str(path), "content_b64": base64.b64encode(raw).decode("ascii"), "size_bytes": len(raw)})

    def _browser_goto(self, body: dict[str, Any]) -> None:
        url = str(body["url"])
        with urllib.request.urlopen(url, timeout=int(body.get("timeout_seconds", 10))) as response:
            raw = response.read()
        text = raw.decode("utf-8", errors="replace")
        status = _response_status(response)
        artifact = None
        if body.get("save_as"):
            artifact_path = _safe_path(str(body["save_as"]))
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(text, encoding="utf-8")
            artifact = str(artifact_path)
        self._json({"url": url, "status": status, "text": text, "artifact": artifact})

    def _browser_check(self, body: dict[str, Any]) -> None:
        result = self._fetch_url(str(body["url"]), int(body.get("timeout_seconds", 10)))
        contains = str(body.get("contains", ""))
        ok = result["status"] < 400 and (not contains or contains in result["text"])
        self._json({"ok": ok, "url": body["url"], "status": result["status"], "contains": contains})

    def _fetch_url(self, url: str, timeout: int) -> dict[str, Any]:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
        return {"status": _response_status(response), "text": text}

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _safe_path(path: str) -> Path:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (WORKSPACE / candidate).resolve()
    try:
        resolved.relative_to(WORKSPACE)
    except ValueError as exc:
        raise ValueError(f"path escapes sandbox workspace: {path}") from exc
    return resolved


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None) or getattr(response, "code", None) or 200
    return int(status)


def main() -> int:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", "18080"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
