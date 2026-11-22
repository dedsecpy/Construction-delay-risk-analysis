from __future__ import annotations

from src.policy.base import Policy, PolicyState


class FixedRankingPolicy(Policy):
    name = "fixed"

    def __init__(self, order: list[str]):
        self.order = list(order)

    def select_next(self, state: PolicyState) -> str | None:
        if state.budget_left <= 0:
            return None
        for f in self.order:
            if f in state.remaining:
                return f
        return None
