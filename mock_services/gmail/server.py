from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


STATE: dict[str, Any] = {"messages": [], "drafts": [], "sent": [], "calls": []}


def _load_messages() -> list[dict[str, Any]]:
    fixture = os.environ.get("GMAIL_FIXTURES")
    if fixture and Path(fixture).exists():
        return json.loads(Path(fixture).read_text(encoding="utf-8"))
    return [
        {
            "id": "msg_101",
            "from": "client@bigcorp.com",
            "subject": "Project delay",
            "body": "Please send an update on the delayed project. We need a clear progress plan.",
            "important": True,
        },
        {
            "id": "msg_102",
            "from": "newsletter@example.com",
            "subject": "Weekly links",
            "body": "No reply needed.",
            "important": False,
        },
    ]


def _reset() -> None:
    STATE["messages"] = _load_messages()
    STATE["drafts"] = []
    STATE["sent"] = []
    STATE["calls"] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/gmail/audit":
            self._json({key: STATE[key] for key in ["calls", "drafts", "sent"]})
        elif self.path == "/health":
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        body = self._read_json()
        if self.path == "/gmail/reset":
            _reset()
            self._json({"status": "reset"})
            return
        if self.path == "/gmail/messages":
            self._record("gmail_list_messages", body)
            max_results = int(body.get("max_results", 20))
            messages = [
                {"id": m["id"], "from": m["from"], "subject": m["subject"], "important": m.get("important", False)}
                for m in STATE["messages"][:max_results]
            ]
            self._json({"messages": messages})
            return
        if self.path == "/gmail/messages/get":
            self._record("gmail_get_message", body)
            message_id = body.get("message_id")
            for message in STATE["messages"]:
                if message["id"] == message_id:
                    self._json({"message": message})
                    return
            self._json({"error": f"message not found: {message_id}"}, status=404)
            return
        if self.path == "/gmail/drafts/save":
            self._record("gmail_save_draft", body)
            draft = {
                "id": f"draft_{len(STATE['drafts']) + 1}",
                "to": body.get("to", ""),
                "subject": body.get("subject", ""),
                "body": body.get("body", ""),
                "reply_to_message_id": body.get("reply_to_message_id"),
            }
            STATE["drafts"].append(draft)
            self._json({"draft": draft})
            return
        if self.path == "/gmail/send":
            self._record("gmail_send_message", body)
            STATE["sent"].append(dict(body))
            self._json({"sent": dict(body)})
            return
        self._json({"error": "not found"}, status=404)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _record(self, tool_name: str, body: dict[str, Any]) -> None:
        STATE["calls"].append({"tool_name": tool_name, "request": dict(body)})

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> int:
    _reset()
    port = int(os.environ.get("PORT", "9100"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
