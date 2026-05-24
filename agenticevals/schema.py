from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class TaskError(ValueError):
    pass


@dataclass(frozen=True)
class WorkspaceSpec:
    fixture_path: str
    setup: list[str] = field(default_factory=list)
    dev_server: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSpec:
    kind: str = "scripted"
    model: str | None = None
    command: str | None = None
    script: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LimitsSpec:
    max_steps: int = 50
    max_minutes: int = 20


@dataclass(frozen=True)
class PolicySpec:
    forbidden_paths: list[str] = field(default_factory=list)
    max_changed_files: int | None = None
    require_changed_files: list[str] = field(default_factory=list)
    allow_network: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolEndpointSpec:
    tool_name: str
    url: str
    method: str = "POST"


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    command: str
    port: int
    health_check: str
    health_check_method: str = "GET"
    ready_timeout: int = 10
    reset_endpoint: str | None = None
    audit_endpoint: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpectedActionSpec:
    service: str
    action_key: str
    required: bool = True
    min_count: int = 1


@dataclass(frozen=True)
class SafetyCheckSpec:
    type: str
    tool_name: str | None = None
    service: str | None = None
    action_key: str | None = None
    max_count: int = 0
    description: str = ""


@dataclass(frozen=True)
class VerifierSpec:
    type: str
    name: str = ""
    weight: float = 1.0
    required: bool = True
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckSpec:
    commands: list[str] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    browser: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ScoreWeights:
    # Point budgets per verifier dimension. Magnitudes are relative: the reward
    # is the weight-normalized average across all emitted criteria.
    command_checks: int = 40
    file_checks: int = 20
    browser_checks: int = 20
    git_policy: int = 20
    expected_actions: int = 25
    audit_safety: int = 25
    tool_dispatch: int = 25
    tool_argument: int = 25
    tool_safety: int = 25


@dataclass(frozen=True)
class TaskSpec:
    id: str
    title: str
    prompt: str
    workspace: WorkspaceSpec
    agent: AgentSpec = field(default_factory=AgentSpec)
    limits: LimitsSpec = field(default_factory=LimitsSpec)
    policies: PolicySpec = field(default_factory=PolicySpec)
    checks: CheckSpec = field(default_factory=CheckSpec)
    score: ScoreWeights = field(default_factory=ScoreWeights)
    services: list[ServiceSpec] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    tool_endpoints: list[ToolEndpointSpec] = field(default_factory=list)
    expected_actions: list[ExpectedActionSpec] = field(default_factory=list)
    safety_checks: list[SafetyCheckSpec] = field(default_factory=list)
    verifiers: list[VerifierSpec] = field(default_factory=list)
    sandbox_files: list[str] = field(default_factory=list)
    sandbox_grader_files: list[str] = field(default_factory=list)
    env_snapshot_commands: list[str] = field(default_factory=list)
    env_snapshot_files: list[str] = field(default_factory=list)
    local_grader_files: list[str] = field(default_factory=list)
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path", None)
        return data

    @classmethod
    def from_file(cls, path: Path) -> "TaskSpec":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TaskError(f"{path} must contain a JSON object")
        task = cls.from_dict(data)
        return TaskSpec(
            id=task.id,
            title=task.title,
            prompt=task.prompt,
            workspace=task.workspace,
            agent=task.agent,
            limits=task.limits,
            policies=task.policies,
            checks=task.checks,
            score=task.score,
            services=task.services,
            tools=task.tools,
            tool_endpoints=task.tool_endpoints,
            expected_actions=task.expected_actions,
            safety_checks=task.safety_checks,
            verifiers=task.verifiers,
            sandbox_files=task.sandbox_files,
            sandbox_grader_files=task.sandbox_grader_files,
            env_snapshot_commands=task.env_snapshot_commands,
            env_snapshot_files=task.env_snapshot_files,
            local_grader_files=task.local_grader_files,
            source_path=path.resolve(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        required = ["id", "title", "prompt", "workspace"]
        missing = [key for key in required if key not in data]
        if missing:
            raise TaskError(f"Missing required task fields: {', '.join(missing)}")
        workspace = WorkspaceSpec(**data["workspace"])
        agent = AgentSpec(**data.get("agent", {}))
        limits = LimitsSpec(**data.get("limits", {}))
        policies = PolicySpec(**data.get("policies", {}))
        checks = CheckSpec(**data.get("checks", {}))
        score = ScoreWeights(**data.get("score", {}))
        services = [ServiceSpec(**item) for item in data.get("services", [])]
        tools = [ToolSpec(**item) for item in data.get("tools", [])]
        tool_endpoints = [ToolEndpointSpec(**item) for item in data.get("tool_endpoints", [])]
        expected_actions = [ExpectedActionSpec(**item) for item in data.get("expected_actions", [])]
        safety_checks = [SafetyCheckSpec(**item) for item in data.get("safety_checks", [])]
        verifiers = [_parse_verifier_spec(item) for item in data.get("verifiers", [])]
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            prompt=str(data["prompt"]),
            workspace=workspace,
            agent=agent,
            limits=limits,
            policies=policies,
            checks=checks,
            score=score,
            services=services,
            tools=tools,
            tool_endpoints=tool_endpoints,
            expected_actions=expected_actions,
            safety_checks=safety_checks,
            verifiers=verifiers,
            sandbox_files=list(data.get("sandbox_files", [])),
            sandbox_grader_files=list(data.get("sandbox_grader_files", [])),
            env_snapshot_commands=list(data.get("env_snapshot_commands", [])),
            env_snapshot_files=list(data.get("env_snapshot_files", [])),
            local_grader_files=list(data.get("local_grader_files", [])),
        )

    def resolve_fixture(self) -> Path:
        raw = Path(self.workspace.fixture_path).expanduser()
        if raw.is_absolute():
            return raw
        base = self.source_path.parent if self.source_path else Path.cwd()
        return (base / raw).resolve()


def _parse_verifier_spec(data: dict[str, Any]) -> VerifierSpec:
    config = dict(data.get("config", {}))
    for key, value in data.items():
        if key not in {"type", "name", "weight", "required", "config"}:
            config[key] = value
    return VerifierSpec(
        type=str(data["type"]),
        name=str(data.get("name", "")),
        weight=float(data.get("weight", 1.0)),
        required=bool(data.get("required", True)),
        config=config,
    )
