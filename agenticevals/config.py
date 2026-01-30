from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    root: Path
    config_root: Path
    task_config_dir: Path
    workspace_path: Path
    runs_path: Path
    traces_path: Path
    env_timeout: int
    action_short_timeout: int
    action_long_timeout: int
    agent_max_steps: int
    model_max_retries: int
    container_type: str
    sandbox_mode: str
    default_agent: str
    default_model: str
    codex_command: str
    claude_command: str

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        repo_root = (root or Path.cwd()).resolve()
        config_root_default = repo_root / "configs"
        task_config_default = config_root_default / "tasks"
        if root is None and not task_config_default.exists():
            bundled = bundled_root()
            config_root_default = bundled / "configs"
            task_config_default = config_root_default / "tasks"
        return cls(
            root=repo_root,
            config_root=_path_env("AGENTICEVALS_CONFIG_ROOT", config_root_default),
            task_config_dir=_path_env("AGENTICEVALS_TASK_CONFIG_DIR", task_config_default),
            workspace_path=_path_env("AGENTICEVALS_WORKSPACE_PATH", repo_root / "workspace"),
            runs_path=_path_env("AGENTICEVALS_RUNS_PATH", repo_root / "runs"),
            traces_path=_path_env("AGENTICEVALS_TRACES_PATH", repo_root / "traces"),
            env_timeout=_int_env("AGENTICEVALS_ENV_TIMEOUT", 10000),
            action_short_timeout=_int_env("AGENTICEVALS_ACTION_SHORT_TIMEOUT", 60),
            action_long_timeout=_int_env("AGENTICEVALS_ACTION_LONG_TIMEOUT", 10000),
            agent_max_steps=_int_env("AGENTICEVALS_AGENT_MAX_STEPS", 50),
            model_max_retries=_int_env("AGENTICEVALS_MODEL_MAX_RETRIES", 3),
            container_type=os.environ.get("AGENTICEVALS_CONTAINER_TYPE", "local"),
            sandbox_mode=os.environ.get("AGENTICEVALS_SANDBOX_MODE", "workspace"),
            default_agent=os.environ.get("AGENTICEVALS_DEFAULT_AGENT", "scripted"),
            default_model=os.environ.get("AGENTICEVALS_DEFAULT_MODEL", ""),
            codex_command=os.environ.get(
                "AGENTICEVALS_CODEX_COMMAND",
                "codex exec --skip-git-repo-check --sandbox workspace-write --cd {workspace} {prompt}",
            ),
            claude_command=os.environ.get(
                "AGENTICEVALS_CLAUDE_COMMAND",
                "claude -p --permission-mode acceptEdits --add-dir={workspace} {prompt}",
            ),
        )

    def ensure_dirs(self) -> None:
        for path in [self.workspace_path, self.runs_path, self.traces_path]:
            path.mkdir(parents=True, exist_ok=True)
        for path in [self.config_root, self.task_config_dir]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)


def bundled_root() -> Path:
    return Path(__file__).resolve().parent / "bundled"
