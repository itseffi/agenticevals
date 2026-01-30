from __future__ import annotations

import html
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin


@dataclass(frozen=True)
class BrowserCheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class BrowserSnapshot:
    url: str
    status: int
    title: str
    text: str
    links: list[dict[str, str]]
    forms: list[dict[str, object]]
    console: list[str] = field(default_factory=list)
    network: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "url": self.url,
            "status": self.status,
            "title": self.title,
            "text": self.text,
            "links": self.links,
            "forms": self.forms,
            "console": self.console,
            "network": self.network,
        }


class _DOMParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.text_parts: list[str] = []
        self.links: list[dict[str, str]] = []
        self.forms: list[dict[str, object]] = []
        self._in_title = False
        self._current_form: dict[str, object] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {key: value or "" for key, value in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "a":
            self.links.append({"href": data.get("href", ""), "text": ""})
        elif tag == "form":
            self._current_form = {"method": data.get("method", "get").lower(), "action": data.get("action", ""), "inputs": []}
            self.forms.append(self._current_form)
        elif tag in {"input", "textarea", "button"} and self._current_form is not None:
            inputs = self._current_form["inputs"]
            assert isinstance(inputs, list)
            inputs.append(
                {
                    "tag": tag,
                    "name": data.get("name", ""),
                    "type": data.get("type", "text"),
                    "value": data.get("value", ""),
                }
            )

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "form":
            self._current_form = None

    def handle_data(self, data: str) -> None:
        stripped = " ".join(data.split())
        if not stripped:
            return
        if self._in_title:
            self.title += stripped
        self.text_parts.append(stripped)
        if self.links:
            self.links[-1]["text"] = (self.links[-1]["text"] + " " + stripped).strip()


class BrowserSession:
    def __init__(self, base_url: str | None = None, timeout: int = 10, artifact_dir: Path | None = None):
        self.base_url = base_url or ""
        self.timeout = timeout
        self.artifact_dir = artifact_dir
        self.current_url = self.base_url
        self.html = ""
        self.status = 0
        self.network: list[dict[str, object]] = []
        self.console: list[str] = []
        self.form_values: dict[str, str] = {}

    def goto(self, url: str) -> BrowserSnapshot:
        target = url if url.startswith(("http://", "https://")) else urljoin(self.base_url, url)
        request = urllib.request.Request(target, headers={"User-Agent": "agenticevals/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                self.status = int(getattr(response, "status", None) or response.getcode() or 200)
                self.current_url = response.geturl()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self.status = exc.code
            self.current_url = target
        self.html = body
        self.network.append({"method": "GET", "url": self.current_url, "status": self.status})
        return self.snapshot()

    def fill(self, name: str, value: str) -> BrowserSnapshot:
        self.form_values[name] = value
        return self.snapshot()

    def click(self, text: str = "", href: str = "", form_index: int | None = None) -> BrowserSnapshot:
        snapshot = self.snapshot()
        if href:
            return self.goto(href)
        if form_index is not None:
            return self.submit(form_index)
        for link in snapshot.links:
            if text and text.lower() in link.get("text", "").lower():
                return self.goto(link.get("href", ""))
        raise ValueError(f"No browser target matched text={text!r} href={href!r} form_index={form_index!r}")

    def submit(self, form_index: int = 0) -> BrowserSnapshot:
        snapshot = self.snapshot()
        form = snapshot.forms[form_index]
        method = str(form.get("method", "get")).lower()
        action = str(form.get("action", ""))
        url = urljoin(self.current_url or self.base_url, action)
        inputs = form.get("inputs", [])
        values: dict[str, str] = {}
        if isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, dict) and item.get("name"):
                    name = str(item["name"])
                    values[name] = self.form_values.get(name, str(item.get("value", "")))
        encoded = urllib.parse.urlencode(values)
        if method == "post":
            data = encoded.encode("utf-8")
            request = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self.html = response.read().decode("utf-8", errors="replace")
                self.status = response.status
                self.current_url = response.url
            self.network.append({"method": "POST", "url": self.current_url, "status": self.status})
        else:
            joiner = "&" if "?" in url else "?"
            self.goto(url + (joiner + encoded if encoded else ""))
        return self.snapshot()

    def snapshot(self) -> BrowserSnapshot:
        parser = _DOMParser()
        parser.feed(self.html)
        text = html.unescape(" ".join(parser.text_parts))
        return BrowserSnapshot(
            url=self.current_url,
            status=self.status,
            title=html.unescape(parser.title),
            text=text,
            links=parser.links,
            forms=parser.forms,
            console=list(self.console),
            network=list(self.network),
        )

    def save_snapshot(self, name: str = "browser-snapshot") -> Path:
        if self.artifact_dir is None:
            raise ValueError("BrowserSession has no artifact directory")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        target = self.artifact_dir / f"{name}.json"
        target.write_text(json.dumps(self.snapshot().to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        html_target = self.artifact_dir / f"{name}.html"
        html_target.write_text(self.html, encoding="utf-8")
        return target


def run_browser_checks(base_url: str | None, checks: list[dict], timeout: int, artifact_dir: Path | None = None) -> list[BrowserCheckResult]:
    results: list[BrowserCheckResult] = []
    browser = BrowserSession(base_url=base_url, timeout=timeout, artifact_dir=artifact_dir)
    for index, check in enumerate(checks, start=1):
        name = check.get("name", f"browser:{index}")
        try:
            if "path" in check or "url" in check:
                snapshot = browser.goto(str(check.get("url") or check.get("path") or "/"))
            else:
                snapshot = browser.snapshot()
            for action in check.get("actions", []):
                kind = action.get("action")
                if kind == "fill":
                    snapshot = browser.fill(str(action["name"]), str(action.get("value", "")))
                elif kind == "click":
                    snapshot = browser.click(
                        text=str(action.get("text", "")),
                        href=str(action.get("href", "")),
                        form_index=action.get("form_index"),
                    )
                elif kind == "submit":
                    snapshot = browser.submit(int(action.get("form_index", 0)))
                elif kind == "goto":
                    snapshot = browser.goto(str(action["url"]))
                else:
                    raise ValueError(f"unknown browser action: {kind}")
            if artifact_dir is not None:
                browser.save_snapshot(f"browser-check-{index}")
        except Exception as exc:
            results.append(BrowserCheckResult(name=name, passed=False, detail=f"browser action failed: {exc}"))
            continue
        failed = _browser_expectation_failure(check, snapshot)
        results.append(BrowserCheckResult(name=name, passed=failed is None, detail=failed or f"status={snapshot.status}"))
    return results


def _browser_expectation_failure(check: dict, snapshot: BrowserSnapshot) -> str | None:
    if "status" in check and int(check["status"]) != snapshot.status:
        return f"expected status {check['status']}, got {snapshot.status}"
    if "contains" in check and str(check["contains"]) not in snapshot.text and str(check["contains"]) not in snapshot.title:
        return f"expected visible text {check['contains']!r}"
    if "not_contains" in check and str(check["not_contains"]) in snapshot.text:
        return f"unexpected visible text {check['not_contains']!r}"
    if "title" in check and str(check["title"]) != snapshot.title:
        return f"expected title {check['title']!r}, got {snapshot.title!r}"
    return None
