from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from agenticevals.computer.context import ComputerContext
from agenticevals.environments import Environment
from agenticevals.rewards import Reward, RewardComponent
from agenticevals.rollouts import AgentResult


class TauRetailEnv(Environment):
    """τ-retail compatible smoke environment.

    The adapter follows the τ-bench retail scoring shape: the agent operates
    over customer/order state and reward is database-state comparison plus
    policy compliance. The bundled items are a smoke set for the harness; use
    an external τ³/τ-retail export for headline benchmark numbers.
    """

    name = "tau-retail"

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self._items: list[dict[str, Any]] = []
        self._index = 0

    def setup(self) -> None:
        self.settings.ensure_dirs()
        root = self.settings.workspace_path / "tau-retail-fixtures"
        root.mkdir(parents=True, exist_ok=True)
        external = Path(str(Path.cwd() / "missing"))
        if "AGENTICEVALS_TAU_RETAIL_TASKS" in os.environ:
            external = Path(os.environ["AGENTICEVALS_TAU_RETAIL_TASKS"]).expanduser().resolve()
        self._items = _load_external_items(external, root) if external.exists() else _bundled_items(root)
        self._index = 0

    def get_next_item(self) -> dict[str, Any] | None:
        if self._index >= len(self._items):
            return None
        item = self._items[self._index]
        self._index += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        policy = (
            "Retail policy: unshipped orders may be canceled; delivered orders may be refunded within 30 days; "
            "orders older than 30 days must not be refunded. Modify db.json only when policy allows it. "
            "Use exact JSON enum values from the existing database schema: status values include processing, canceled, delivered; "
            "refund values include none, issued."
        )
        return f"{policy}\n\nCustomer request:\n{item['request']}\n\nReturn a concise final response."

    def fixture_path(self, item: dict[str, Any]) -> Path:
        return Path(str(item["fixture_path"]))

    def compute_reward(self, item: dict[str, Any], result: AgentResult, ctx: ComputerContext) -> Reward:
        actual = json.loads(ctx.read_file("db.json"))
        expected = item["expected_db"]
        db_passed = actual == expected
        policy_passed = _policy_passed(item, result, actual)
        components = [
            RewardComponent("database_state", 1.0 if db_passed else 0.0, 1.0, db_passed, _db_detail(actual, expected)),
            RewardComponent("policy_compliance", 1.0 if policy_passed else 0.0, 1.0, policy_passed, str(item.get("policy_expectation", ""))),
        ]
        return Reward(
            value=sum(component.value for component in components),
            max_value=sum(component.max_value for component in components),
            passed=all(component.passed for component in components),
            components=components,
        )


def _bundled_items(root: Path) -> list[dict[str, Any]]:
    seeds = [
        {
            "id": "cancel-unshipped-order",
            "request": "Cancel order O-100 for customer C-1.",
            "db": {"orders": [{"id": "O-100", "customer_id": "C-1", "status": "processing", "days_since_delivery": None, "refund": "none"}]},
            "expected_db": {"orders": [{"id": "O-100", "customer_id": "C-1", "status": "canceled", "days_since_delivery": None, "refund": "none"}]},
            "final": "Canceled order O-100.",
        },
        {
            "id": "refund-recent-delivery",
            "request": "Refund delivered order O-200 for customer C-2.",
            "db": {"orders": [{"id": "O-200", "customer_id": "C-2", "status": "delivered", "days_since_delivery": 12, "refund": "none"}]},
            "expected_db": {"orders": [{"id": "O-200", "customer_id": "C-2", "status": "delivered", "days_since_delivery": 12, "refund": "issued"}]},
            "final": "Issued the refund for order O-200.",
        },
        {
            "id": "deny-late-refund",
            "request": "Refund delivered order O-300 for customer C-3.",
            "db": {"orders": [{"id": "O-300", "customer_id": "C-3", "status": "delivered", "days_since_delivery": 45, "refund": "none"}]},
            "expected_db": {"orders": [{"id": "O-300", "customer_id": "C-3", "status": "delivered", "days_since_delivery": 45, "refund": "none"}]},
            "final": "I cannot issue that refund because the order is outside the 30 day window.",
            "policy_expectation": "must deny late refund",
        },
    ]
    items = []
    for seed in seeds:
        fixture = root / seed["id"]
        fixture.mkdir(parents=True, exist_ok=True)
        (fixture / "db.json").write_text(json.dumps(seed["db"], indent=2, sort_keys=True), encoding="utf-8")
        items.append(
            {
                **seed,
                "fixture_path": str(fixture),
                "script": [
                    {"action": "write_file", "path": "db.json", "content": json.dumps(seed["expected_db"], indent=2, sort_keys=True) + "\n"},
                    {"action": "final", "message": seed["final"]},
                ],
            }
        )
    return items


def _load_external_items(path: Path, root: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_items = data.get("tasks", data if isinstance(data, list) else [])
    items = []
    for raw in raw_items:
        task_id = str(raw["id"])
        fixture = root / task_id
        fixture.mkdir(parents=True, exist_ok=True)
        db = raw.get("db") or raw.get("initial_state") or {}
        expected = raw.get("expected_db") or raw.get("goal_state") or deepcopy(db)
        (fixture / "db.json").write_text(json.dumps(db, indent=2, sort_keys=True), encoding="utf-8")
        items.append(
            {
                "id": task_id,
                "request": str(raw.get("request") or raw.get("instruction") or raw.get("user_goal") or ""),
                "db": db,
                "expected_db": expected,
                "fixture_path": str(fixture),
                "script": raw.get("script", []),
                "policy_expectation": raw.get("policy_expectation", ""),
            }
        )
    return items


def _policy_passed(item: dict[str, Any], result: AgentResult, actual: dict[str, Any]) -> bool:
    expectation = str(item.get("policy_expectation", ""))
    if expectation == "must deny late refund":
        response = result.final_response.lower()
        denied = any(marker in response for marker in ("cannot", "can't", "outside", "exceed", "denied", "deny"))
        return denied and actual == item["expected_db"]
    return actual == item["expected_db"]


def _db_detail(actual: dict[str, Any], expected: dict[str, Any]) -> str:
    if actual == expected:
        return "database state matches expected"
    return f"actual={actual}; expected={expected}"
