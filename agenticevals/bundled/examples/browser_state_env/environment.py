from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticevals.computer.context import ComputerContext
from agenticevals.environments import Environment
from agenticevals.rewards import Reward, RewardComponent
from agenticevals.rollouts import AgentResult


class BrowserStateEnv(Environment):
    name = "browser-state"

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self._items = [
            {
                "id": "visible-ready-state",
                "title": "Patch browser-visible ready state",
                "script": [
                    {
                        "action": "replace",
                        "path": "index.html",
                        "old": "Dashboard loading",
                        "new": "Dashboard ready",
                    },
                    {
                        "action": "replace",
                        "path": "index.html",
                        "old": "Waiting for agent patch.",
                        "new": "Browser verifier can see this state.",
                    },
                    {"action": "final", "message": "Updated browser-visible copy."},
                ],
            }
        ]
        self._index = 0

    def setup(self) -> None:
        self._index = 0

    def get_next_item(self) -> dict[str, Any] | None:
        if self._index >= len(self._items):
            return None
        item = self._items[self._index]
        self._index += 1
        return item

    def format_prompt(self, item: dict[str, Any]) -> str:
        return "Patch the local web page so a browser sees the dashboard ready state."

    def fixture_path(self, item: dict[str, Any]) -> Path:
        return Path(__file__).parent / "fixture"

    def compute_reward(self, item: dict[str, Any], result: AgentResult, ctx: ComputerContext) -> Reward:
        browser = ctx.browser(base_url=ctx.workspace.as_uri() + "/", timeout=5)
        snapshot = browser.goto("index.html")
        artifact = browser.save_snapshot("reward-browser-state")
        visible = "Dashboard ready" in snapshot.text and "Browser verifier can see this state." in snapshot.text
        status_ok = snapshot.status == 200
        return Reward(
            value=1.0 if visible and status_ok else 0.0,
            max_value=1.0,
            passed=visible and status_ok,
            components=[
                RewardComponent("browser_status", 1.0 if status_ok else 0.0, 1.0, status_ok, f"status={snapshot.status}"),
                RewardComponent("browser_text", 1.0 if visible else 0.0, 1.0, visible, f"snapshot={artifact}"),
            ],
        )
