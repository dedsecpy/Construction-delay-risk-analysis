"""Policy interface for sequential inspection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.bn.inference import DelayRiskBN


@dataclass
class PolicyState:
    bn: DelayRiskBN
    remaining: set[str]
    budget_left: int
    history: list[str] = field(default_factory=list)


class Policy(ABC):
    name: str = "base"

    @abstractmethod
    def select_next(self, state: PolicyState) -> str | None:
        ...

    def update(self, state: PolicyState, factor: str, present: bool) -> PolicyState:
        state.bn.observe(factor, present)
        state.remaining.discard(factor)
        state.budget_left -= 1
        state.history.append(factor)
        return state
