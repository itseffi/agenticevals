from __future__ import annotations

import importlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agenticevals.agents.factory import create_agent
from agenticevals.backends import create_backend
from agenticevals.computer.context import ComputerContext
from agenticevals.config import Settings
from agenticevals.rewards import Reward
from agenticevals.rollouts import AgentResult, EvalResult, RolloutResult
from agenticevals.rollouts.types import now
from agenticevals.run_store import EvalRunStore
from agenticevals.schema import AgentSpec, LimitsSpec, TaskSpec, WorkspaceSpec
from agenticevals.trace import Trajectory
from agenticevals.trajectory_export import write_typed_trajectory


class EnvironmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnvironmentOptions:
    max_items: int | None = None
    agent: str | None = None
    max_minutes: int = 20
    resume: Path | None = None
    backend: str = "local"
    image: str | None = None


class Environment:
    name = "environment"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()

    def setup(self) -> None:
        pass

    def get_next_item(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def format_prompt(self, item: dict[str, Any]) -> str:
        raise NotImplementedError

    def fixture_path(self, item: dict[str, Any]) -> Path:
        raise NotImplementedError

    def compute_reward(self, item: dict[str, Any], result: AgentResult, ctx: ComputerContext) -> Reward:
        raise NotImplementedError

    def make_task_spec(self, item: dict[str, Any], prompt: str, agent_kind: str) -> TaskSpec:
        return TaskSpec(
            id=f"{self.name}-{item['id']}",
            title=str(item.get("title", item["id"])),
            prompt=prompt,
            workspace=WorkspaceSpec(fixture_path=str(self.fixture_path(item))),
            agent=AgentSpec(kind=agent_kind, script=item.get("script", [])),
            limits=LimitsSpec(max_minutes=20, max_steps=int(item.get("max_steps", 50))),
        )

    def rollout(
        self,
        item: dict[str, Any],
        options: EnvironmentOptions | None = None,
        eval_store: EvalRunStore | None = None,
    ) -> RolloutResult:
        opts = options or EnvironmentOptions()
        agent_kind = opts.agent or self.settings.default_agent
        prompt = self.format_prompt(item)
        task = self.make_task_spec(item, prompt, agent_kind)
        trace = Trajectory(task_id=task.id)
        run_id = trace.run_id
        backend = create_backend(opts.backend, image=opts.image)
        if eval_store is not None:
            run_dir = eval_store.rollout_dir(str(item["id"]), run_id)
        else:
            stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            run_dir = self.settings.runs_path / f"{task.id}-{stamp}"
            run_dir.mkdir(parents=True, exist_ok=False)
        backend_workspace = backend.create_workspace(self.fixture_path(item), run_dir)
        started = now()
        trace.add("environment.rollout.start", environment=self.name, item_id=item["id"], agent=agent_kind)
        ctx = ComputerContext(
            backend=backend,
            backend_workspace=backend_workspace,
            trace=trace,
            default_timeout=self.settings.action_long_timeout,
        )

        agent = create_agent(task, self.settings, override=agent_kind)
        raw_result = agent.run(
            task,
            backend_workspace.workspace,
            trace,
            timeout=max(1, opts.max_minutes) * 60,
            computer=ctx,
        )
        agent_result = AgentResult(
            ok=raw_result.ok,
            final_response=raw_result.final_message,
            metadata=raw_result.metadata,
        )
        reward = self.compute_reward(item, agent_result, ctx)
        status = "passed" if reward.passed else "failed"
        completed = now()

        diff_path = backend_workspace.run_dir / "diff.patch"
        diff_path.write_text(ctx.diff(), encoding="utf-8")
        rollout = RolloutResult(
            run_id=run_id,
            environment=self.name,
            item_id=str(item["id"]),
            agent=agent_kind,
            status=status,
            workspace=backend_workspace.workspace,
            run_dir=backend_workspace.run_dir,
            started_at=started,
            completed_at=completed,
            prompt=prompt,
            agent_result=agent_result,
            reward=reward,
            artifacts={"diff": str(diff_path)},
        )
        trace.add("environment.rollout.finish", **rollout.to_dict())
        trace.write_jsonl(backend_workspace.run_dir / "trajectory.jsonl")
        write_typed_trajectory(
            trace,
            backend_workspace.run_dir / "trajectory.json",
            task=task,
            metadata={"environment": self.name, "item_id": str(item["id"])},
        )
        (backend_workspace.run_dir / "rollout.json").write_text(json.dumps(rollout.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        (backend_workspace.run_dir / "item.json").write_text(
            json.dumps(
                {"schema_version": "agenticevals.environment-item.v1", "environment": self.name, "item": item},
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        backend.cleanup(backend_workspace)
        return rollout

    def evaluate(self, options: EnvironmentOptions | None = None) -> EvalResult:
        opts = options or EnvironmentOptions()
        self.setup()
        started = now()
        eval_store = EvalRunStore.resume(opts.resume) if opts.resume else EvalRunStore.create(self.settings.runs_path, self.name)
        completed_ids = eval_store.completed_item_ids()
        rollouts: list[RolloutResult] = eval_store.load_rollouts()
        processed = 0
        while True:
            if opts.max_items is not None and processed >= opts.max_items:
                break
            item = self.get_next_item()
            if item is None:
                break
            if str(item["id"]) in completed_ids:
                continue
            rollouts.append(self.rollout(item, opts, eval_store=eval_store))
            processed += 1
        completed = now()
        result = EvalResult(
            environment=self.name,
            agent=opts.agent or self.settings.default_agent,
            run_dir=eval_store.path,
            started_at=started,
            completed_at=completed,
            rollouts=rollouts,
        )
        eval_store.write_eval(result.to_dict())
        return result


def load_environment(spec: str, settings: Settings | None = None) -> Environment:
    if ":" not in spec:
        raise EnvironmentError("Environment spec must be '<module>:<ClassName>'")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    env = cls(settings=settings)
    if not isinstance(env, Environment):
        raise EnvironmentError(f"{spec} did not produce an Environment")
    return env
