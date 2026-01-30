from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from agenticevals.config import Settings
from agenticevals.environments import EnvironmentOptions, load_environment
from agenticevals.stats import bootstrap_ci


def run_environment_baselines(
    environment: str,
    settings: Settings,
    *,
    agents: list[str],
    max_items: int | None = None,
    trials: int = 1,
    backend: str = "local",
    image: str | None = None,
    max_minutes: int = 20,
    output: Path | None = None,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    run_dir = output.expanduser().resolve() if output else settings.runs_path / f"env-baselines-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for agent in agents:
        trial_summaries = []
        for trial_index in range(trials):
            env = load_environment(environment, settings=settings)
            result = env.evaluate(
                EnvironmentOptions(
                    max_items=max_items,
                    agent=agent,
                    max_minutes=max_minutes,
                    backend=backend,
                    image=image,
                )
            )
            summary = result.to_dict()
            summary["trial_index"] = trial_index
            trial_summaries.append(summary)
        rows.append(_summarize_agent(agent, trial_summaries, trials))

    payload = {
        "schema_version": "agenticevals.environment-baselines.v1",
        "environment": environment,
        "suite": environment,
        "backend": backend,
        "image": image,
        "max_items": max_items,
        "trials": trials,
        "agents": agents,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": sorted(rows, key=lambda row: (-row["pass_at_1"], -row["pass_power_k"], row["agent"])),
    }
    (run_dir / "baselines.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "RESULTS.md").write_text(_results_markdown(payload), encoding="utf-8")
    payload["run_dir"] = str(run_dir)
    return payload


def _summarize_agent(agent: str, trial_summaries: list[dict[str, Any]], trials: int) -> dict[str, Any]:
    rollouts = []
    for summary in trial_summaries:
        for rollout in summary.get("rollouts", []):
            rollouts.append({**rollout, "trial_index": summary["trial_index"], "eval_run_dir": summary["run_dir"]})
    by_item: dict[str, list[dict[str, Any]]] = {}
    for rollout in rollouts:
        by_item.setdefault(str(rollout["item_id"]), []).append(rollout)

    first_attempts = []
    pass_power_items = []
    for item_rollouts in by_item.values():
        ordered = sorted(item_rollouts, key=lambda row: int(row["trial_index"]))
        first_attempts.append(_passed(ordered[0]))
        pass_power_items.append(all(_passed(row) for row in ordered[:trials]) and len(ordered) >= trials)

    passed_rollouts = [_passed(row) for row in rollouts]
    reward_rates = [_reward_rate(row) for row in rollouts]
    costs = [_cost_usd(row) for row in rollouts]
    known_costs = [value for value in costs if value is not None]
    successes = sum(1 for row in rollouts if _passed(row))
    total_cost = sum(known_costs) if known_costs else None
    return {
        "agent": agent,
        "items": len(by_item),
        "trials": trials,
        "rollouts": len(rollouts),
        "passed": sum(1 for value in passed_rollouts if value),
        "pass_rate": _mean(passed_rollouts),
        "pass_rate_ci": bootstrap_ci(passed_rollouts, seed=_seed(agent, "pass_rate")),
        "pass_at_1": _mean(first_attempts),
        "pass_at_1_ci": bootstrap_ci(first_attempts, seed=_seed(agent, "pass_at_1")),
        "pass_power_k": _mean(pass_power_items),
        "pass_power_k_ci": bootstrap_ci(pass_power_items, seed=_seed(agent, "pass_power_k")),
        "mean_reward_rate": _mean(reward_rates),
        "mean_reward_rate_ci": bootstrap_ci(reward_rates, seed=_seed(agent, "reward_rate")),
        "total_cost_usd": round(total_cost, 8) if total_cost is not None else None,
        "cost_per_success_usd": round(total_cost / successes, 8) if total_cost is not None and successes else None,
        "mean_duration_seconds": _mean([float(row.get("duration_seconds", 0.0) or 0.0) for row in rollouts]),
        "eval_run_dirs": [str(summary["run_dir"]) for summary in trial_summaries],
    }


def _results_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Environment Baseline Results",
        "",
        f"Environment: `{payload['environment']}`",
        f"Trials: `{payload['trials']}`",
        "",
        "| Agent | Items | Trials | Pass@1 | Pass^k | 95% CI | Cost/success | Mean seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        ci = row["pass_at_1_ci"]
        cost = "n/a" if row["cost_per_success_usd"] is None else f"${row['cost_per_success_usd']:.4f}"
        lines.append(
            f"| {row['agent']} | {row['items']} | {row['trials']} | {row['pass_at_1']:.3f} | "
            f"{row['pass_power_k']:.3f} | [{ci['low']:.3f}, {ci['high']:.3f}] | {cost} | "
            f"{row['mean_duration_seconds']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def _passed(rollout: dict[str, Any]) -> bool:
    reward = rollout.get("reward", {})
    return bool(reward.get("passed", rollout.get("status") == "passed"))


def _reward_rate(rollout: dict[str, Any]) -> float:
    reward = rollout.get("reward", {})
    value = float(reward.get("value", 0.0) or 0.0)
    max_value = float(reward.get("max_value", 0.0) or 0.0)
    return value / max_value if max_value else 0.0


def _cost_usd(rollout: dict[str, Any]) -> float | None:
    metadata = rollout.get("agent_result", {}).get("metadata", {})
    candidates = [metadata.get("cost_usd")]
    usage = metadata.get("usage")
    if isinstance(usage, dict):
        candidates.append(usage.get("cost_usd"))
    for candidate in candidates:
        if candidate is not None:
            return float(candidate)
    return None


def _mean(values: list[float | bool]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _seed(*parts: str) -> int:
    return sum(ord(char) for part in parts for char in part)
