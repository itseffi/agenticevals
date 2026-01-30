from __future__ import annotations

from .base import BaseAgent
from .claude_loop import ClaudeAgent
from .command import CommandAgent
from .direct_model import DirectModelAgent
from .http_agent import HTTPAgent, endpoint_from_env_or_task
from .model_loop import ModelLoopAgent
from .native_tools import GeminiNativeAgent, OpenAINativeAgent
from .noop import NoopAgent
from .scripted import ScriptedAgent
from agenticevals.config import Settings
from agenticevals.schema import TaskSpec


def create_agent(task: TaskSpec, settings: Settings, override: str | None = None) -> BaseAgent:
    kind = override or task.agent.kind or settings.default_agent
    if kind == "noop":
        return NoopAgent()
    if kind == "scripted":
        return ScriptedAgent()
    if kind == "command":
        if not task.agent.command:
            raise ValueError("agent.kind=command requires agent.command in task config")
        return CommandAgent(task.agent.command, name="command")
    if kind == "codex":
        return CommandAgent(task.agent.command or settings.codex_command, name="codex")
    if kind == "claude-code":
        return CommandAgent(task.agent.command or settings.claude_command, name="claude-code")
    if kind == "direct-model":
        return DirectModelAgent(provider="openai", model=task.agent.model or settings.default_model)
    if kind in {"openai", "openai-api"}:
        return OpenAINativeAgent()
    if kind in {"gemini", "gemini-api"}:
        return GeminiNativeAgent()
    if kind == "model-loop":
        return ModelLoopAgent()
    if kind in {"claude", "claude-api"}:
        return ClaudeAgent()
    if kind == "http":
        return HTTPAgent(endpoint_from_env_or_task(task))
    raise ValueError(f"Unknown agent kind: {kind}")


def adapter_status(settings: Settings) -> list[dict[str, str]]:
    return [
        {"name": "noop", "kind": "builtin", "status": "available"},
        {"name": "scripted", "kind": "builtin", "status": "available"},
        {"name": "command", "kind": "external-cli", "status": "uses .agenticevals task/result/trace files"},
        {"name": "codex", "kind": "external-cli", "status": f"configured: {settings.codex_command}"},
        {"name": "claude-code", "kind": "external-cli", "status": f"configured: {settings.claude_command}"},
        {
            "name": "direct-model",
            "kind": "api",
            "status": "requires OPENAI_API_KEY and AGENTICEVALS_DEFAULT_MODEL",
        },
        {
            "name": "openai",
            "kind": "api-or-fixture",
            "status": "OpenAI Responses API with native function calls; requires OPENAI_API_KEY",
        },
        {
            "name": "gemini",
            "kind": "api-or-fixture",
            "status": "Gemini generateContent with native function calls; requires GEMINI_API_KEY or GOOGLE_API_KEY",
        },
        {"name": "model-loop", "kind": "api-or-fixture", "status": "multi-turn tool loop with parser registry"},
        {
            "name": "claude",
            "kind": "api-or-fixture",
            "status": "anthropic messages API with native tool_use blocks; requires ANTHROPIC_API_KEY",
        },
        {"name": "http", "kind": "external-http", "status": "requires task.agent.command or AGENTICEVALS_HTTP_AGENT_URL"},
    ]
