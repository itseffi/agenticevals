from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar


PRICES_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    "claude-opus-4-7": {"input": 0.015, "output": 0.075, "cache_write": 0.01875, "cache_read": 0.0015},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015, "cache_write": 0.00375, "cache_read": 0.0003},
    "claude-haiku-4-5": {"input": 0.0008, "output": 0.004, "cache_write": 0.001, "cache_read": 0.00008},
}


ANTHROPIC_MODEL_FAMILIES = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}


@dataclass(frozen=True)
class ModelResponse:
    text: str
    provider: str
    model: str
    cached: bool
    usage: dict[str, Any]
    cost_usd: float


@dataclass(frozen=True)
class NativeToolResponse:
    text: str
    tool_calls: list[dict[str, Any]]
    provider: str
    model: str
    cached: bool
    usage: dict[str, Any]
    cost_usd: float
    latency_ms: float
    stop_reason: str = ""
    raw: dict[str, Any] | None = None


_T = TypeVar("_T")


def _is_retryable(exc: BaseException) -> bool:
    """Transient API failures worth retrying: rate limits, server errors, network.

    Client errors (4xx other than 429) are deterministic and not retried.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    if isinstance(exc, urllib.error.URLError):
        return True
    return isinstance(exc, TimeoutError)


def _retry_attempts() -> int:
    try:
        return max(1, int(os.environ.get("AGENTICEVALS_MODEL_MAX_RETRIES", "3")))
    except ValueError:
        return 3


def _with_retries(fn: Callable[[], _T], *, attempts: int, sleep: Callable[[float], None] = time.sleep) -> _T:
    """Call `fn`, retrying transient failures with exponential backoff.

    `attempts` is the total number of tries. Non-retryable errors propagate
    immediately so a malformed request is not hammered.
    """
    last: BaseException | None = None
    for index in range(max(1, attempts)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below if not retryable
            if not _is_retryable(exc) or index == attempts - 1:
                raise
            last = exc
            sleep(min(2.0**index * 0.5, 8.0))
    assert last is not None  # unreachable: loop either returns or raises
    raise last


def _gemini_request(model: str, body: dict[str, Any], api_key: str) -> urllib.request.Request:
    # Pass the key as a header, not a URL query param: query strings leak into
    # proxy/access logs and crash dumps.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    return urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )


def _urlopen_json(request: urllib.request.Request, timeout: int) -> dict[str, Any]:
    def _call() -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    return _with_retries(_call, attempts=_retry_attempts())


def _cache_key(provider: str, model: str, payload: Any, params: dict[str, Any] | None = None) -> str:
    """Stable cache key for a model call.

    Includes `params` (e.g. sampling settings) so that two calls with the same
    prompt but different parameters never collide on a cached response.
    """
    material = {"provider": provider, "model": model, "payload": payload, "params": params or {}}
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()


class RateLimiter:
    def __init__(self, min_interval_seconds: float):
        self.min_interval_seconds = min_interval_seconds
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last = time.monotonic()


def openai_response(
    model: str,
    input_text: str,
    *,
    timeout: int,
    cache_dir: Path | None = None,
    params: dict[str, Any] | None = None,
) -> ModelResponse:
    cache = cache_dir or Path(os.environ.get("AGENTICEVALS_CACHE_DIR", ".cache/agenticevals")).expanduser()
    use_cache = os.environ.get("AGENTICEVALS_USE_CACHE", "true").lower() != "false"
    cache.mkdir(parents=True, exist_ok=True)
    key = _cache_key("openai", model, input_text, params)
    cache_path = cache / f"{key}.json"
    if use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return ModelResponse(text=payload["text"], provider="openai", model=model, cached=True, usage=payload.get("usage", {}), cost_usd=0.0)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    rpm = float(os.environ.get("AGENTICEVALS_MIN_REQUEST_INTERVAL_SECONDS", "0"))
    if rpm > 0:
        RateLimiter(rpm).wait()
    body = json.dumps({"model": model, "input": input_text}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    raw = _urlopen_json(request, timeout)
    text = _extract_response_text(raw)
    usage = raw.get("usage", {})
    cost = estimate_cost(model, usage)
    cache_path.write_text(json.dumps({"text": text, "usage": usage, "cost_usd": cost}, indent=2, sort_keys=True), encoding="utf-8")
    return ModelResponse(text=text, provider="openai", model=model, cached=False, usage=usage, cost_usd=cost)


def openai_responses_native(
    model: str,
    input_items: list[dict[str, Any]],
    *,
    timeout: int,
    tools: list[dict[str, Any]] | None = None,
    previous_response_id: str | None = None,
    cache_dir: Path | None = None,
    fixture_provider: Any | None = None,
) -> NativeToolResponse:
    started = time.monotonic()
    cache = cache_dir or Path(os.environ.get("AGENTICEVALS_CACHE_DIR", ".cache/agenticevals")).expanduser()
    use_cache = os.environ.get("AGENTICEVALS_USE_CACHE", "true").lower() != "false"
    cache.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {"model": model, "input": input_items}
    if tools:
        body["tools"] = tools
    if previous_response_id:
        body["previous_response_id"] = previous_response_id
    key = hashlib.sha256(json.dumps({"provider": "openai-native", **body}, sort_keys=True, default=str).encode()).hexdigest()
    cache_path = cache / f"openai-native-{key}.json"
    if fixture_provider is None and use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return _build_openai_native_response(payload, model=model, cached=True, latency_ms=(time.monotonic() - started) * 1000)
    if fixture_provider is not None:
        raw = fixture_provider(input_items, tools or [], previous_response_id)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        rpm = float(os.environ.get("AGENTICEVALS_MIN_REQUEST_INTERVAL_SECONDS", "0"))
        if rpm > 0:
            RateLimiter(rpm).wait()
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        raw = _urlopen_json(request, timeout)
    latency_ms = (time.monotonic() - started) * 1000
    if fixture_provider is None:
        cache_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    return _build_openai_native_response(raw, model=model, cached=False, latency_ms=latency_ms)


def gemini_generate_content(
    model: str,
    contents: list[dict[str, Any]],
    *,
    timeout: int,
    tools: list[dict[str, Any]] | None = None,
    cache_dir: Path | None = None,
    fixture_provider: Any | None = None,
) -> NativeToolResponse:
    started = time.monotonic()
    cache = cache_dir or Path(os.environ.get("AGENTICEVALS_CACHE_DIR", ".cache/agenticevals")).expanduser()
    use_cache = os.environ.get("AGENTICEVALS_USE_CACHE", "true").lower() != "false"
    cache.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {"contents": contents}
    if tools:
        body["tools"] = [{"functionDeclarations": tools}]
    key = hashlib.sha256(json.dumps({"provider": "gemini", "model": model, **body}, sort_keys=True, default=str).encode()).hexdigest()
    cache_path = cache / f"gemini-{key}.json"
    if fixture_provider is None and use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return _build_gemini_response(payload, model=model, cached=True, latency_ms=(time.monotonic() - started) * 1000)
    if fixture_provider is not None:
        raw = fixture_provider(contents, tools or [])
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
        rpm = float(os.environ.get("AGENTICEVALS_MIN_REQUEST_INTERVAL_SECONDS", "0"))
        if rpm > 0:
            RateLimiter(rpm).wait()
        request = _gemini_request(model, body, api_key)
        raw = _urlopen_json(request, timeout)
    latency_ms = (time.monotonic() - started) * 1000
    if fixture_provider is None:
        cache_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    return _build_gemini_response(raw, model=model, cached=False, latency_ms=latency_ms)


def estimate_cost(model: str, usage: dict[str, Any]) -> float:
    price = PRICES_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    input_tokens = float(usage.get("input_tokens", usage.get("prompt_tokens", usage.get("promptTokenCount", 0))) or 0)
    output_tokens = float(usage.get("output_tokens", usage.get("completion_tokens", usage.get("candidatesTokenCount", 0))) or 0)
    cache_write_tokens = float(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read_tokens = float(usage.get("cache_read_input_tokens", 0) or 0)
    cost = (input_tokens / 1000.0) * price.get("input", 0.0) + (output_tokens / 1000.0) * price.get("output", 0.0)
    cost += (cache_write_tokens / 1000.0) * price.get("cache_write", price.get("input", 0.0))
    cost += (cache_read_tokens / 1000.0) * price.get("cache_read", price.get("input", 0.0))
    return cost


@dataclass(frozen=True)
class AnthropicMessage:
    """A single Anthropic Messages API response.

    Preserves the structured assistant content blocks so callers can act on
    tool_use blocks without parsing free-form text.
    """

    text: str
    content_blocks: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    stop_reason: str
    provider: str
    model: str
    cached: bool
    usage: dict[str, Any]
    cost_usd: float
    latency_ms: float


def anthropic_messages(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int,
    tools: list[dict[str, Any]] | None = None,
    system: str | None = None,
    max_tokens: int = 4096,
    cache_dir: Path | None = None,
    fixture_provider: Any | None = None,
) -> AnthropicMessage:
    """Call the Anthropic Messages API with native tool calling.

    Returns the structured assistant turn (text + tool_use blocks) plus usage and cost.
    If `fixture_provider` is given, it is called instead of the real API — used for
    tests and deterministic eval fixtures. The fixture provider takes
    (messages, tools, system) and returns a dict that mirrors the Anthropic
    response shape.
    """
    started = time.monotonic()
    cache = cache_dir or Path(os.environ.get("AGENTICEVALS_CACHE_DIR", ".cache/agenticevals")).expanduser()
    use_cache = os.environ.get("AGENTICEVALS_USE_CACHE", "true").lower() != "false"
    cache.mkdir(parents=True, exist_ok=True)
    key_payload = {
        "provider": "anthropic",
        "model": model,
        "messages": messages,
        "tools": tools or [],
        "system": system or "",
        "max_tokens": max_tokens,
    }
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, default=str).encode()).hexdigest()
    cache_path = cache / f"anthropic-{key}.json"

    if fixture_provider is None and use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        latency_ms = (time.monotonic() - started) * 1000
        return _build_anthropic_message(payload, model=model, cached=True, latency_ms=latency_ms)

    if fixture_provider is not None:
        raw = fixture_provider(messages, tools or [], system)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the claude adapter")
        rpm = float(os.environ.get("AGENTICEVALS_MIN_REQUEST_INTERVAL_SECONDS", "0"))
        if rpm > 0:
            RateLimiter(rpm).wait()
        body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if tools:
            body["tools"] = tools
        if system:
            body["system"] = system
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        raw = _urlopen_json(request, timeout)

    latency_ms = (time.monotonic() - started) * 1000
    if fixture_provider is None:
        cache_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    return _build_anthropic_message(raw, model=model, cached=False, latency_ms=latency_ms)


def _build_anthropic_message(payload: dict[str, Any], *, model: str, cached: bool, latency_ms: float) -> AnthropicMessage:
    content = payload.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id", "")),
                    "name": str(block.get("name", "")),
                    "input": block.get("input", {}) or {},
                }
            )
    usage = payload.get("usage", {}) or {}
    cost = estimate_cost(model, usage)
    return AnthropicMessage(
        text="\n".join(part for part in text_parts if part).strip(),
        content_blocks=list(content),
        tool_calls=tool_calls,
        stop_reason=str(payload.get("stop_reason", "")),
        provider="anthropic",
        model=model,
        cached=cached,
        usage=usage,
        cost_usd=cost,
        latency_ms=latency_ms,
    )


def _extract_response_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(str(content.get("text", "")))
    return "\n".join(part for part in chunks if part).strip()


def _build_openai_native_response(payload: dict[str, Any], *, model: str, cached: bool, latency_ms: float) -> NativeToolResponse:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in payload.get("output", []):
        item_type = item.get("type")
        if item_type in {"function_call", "tool_call"}:
            raw_arguments = item.get("arguments", {})
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": raw_arguments}
            else:
                arguments = raw_arguments or {}
            tool_calls.append(
                {
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "name": str(item.get("name", "")),
                    "input": arguments,
                }
            )
        elif item_type == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    text_parts.append(str(content.get("text", "")))
        elif item_type != "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    text_parts.append(str(content.get("text", "")))
    usage = payload.get("usage", {}) or {}
    return NativeToolResponse(
        text="\n".join(part for part in text_parts if part).strip(),
        tool_calls=tool_calls,
        provider="openai",
        model=model,
        cached=cached,
        usage=usage,
        cost_usd=estimate_cost(model, usage),
        latency_ms=latency_ms,
        stop_reason=str(payload.get("status", "")),
        raw=payload,
    )


def _build_gemini_response(payload: dict[str, Any], *, model: str, cached: bool, latency_ms: float) -> NativeToolResponse:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                text_parts.append(str(part["text"]))
            if "functionCall" in part:
                call = part["functionCall"] or {}
                name = str(call.get("name", ""))
                tool_calls.append(
                    {
                        "id": str(call.get("id") or f"gemini-{len(tool_calls) + 1}"),
                        "name": name,
                        "input": call.get("args") or {},
                    }
                )
    usage = payload.get("usageMetadata", {}) or {}
    normalized_usage = {
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0),
        **usage,
    }
    return NativeToolResponse(
        text="\n".join(part for part in text_parts if part).strip(),
        tool_calls=tool_calls,
        provider="gemini",
        model=model,
        cached=cached,
        usage=normalized_usage,
        cost_usd=estimate_cost(model, normalized_usage),
        latency_ms=latency_ms,
        stop_reason=str(payload.get("promptFeedback", {}).get("blockReason", "")),
        raw=payload,
    )
