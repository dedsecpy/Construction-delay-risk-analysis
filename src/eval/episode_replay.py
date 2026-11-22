"""Replay policies on NYC episodes under budget B."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.bn.inference import DelayRiskBN
from src.policy.base import Policy, PolicyState
from src.policy.utility import should_alert


@dataclass
class Trace:
    pid: str
    policy: str
    budget: int
    severe_delay: int
    alerted: bool
    p_severe: float
    inspections: list[str]
    steps_to_alert: int | None


def replay_episode(
    policy: Policy,
    bn: DelayRiskBN,
    episode: pd.DataFrame,
    budget: int,
    all_factors: list[str],
    threshold: float = 0.65,
) -> Trace:
    """Replay one project under a budgeted inspection policy.

    Semantics:
    - Reasons that never appear for the PID are treated as known-absent.
    - Only factors that appear in the episode are inspectable.
    - An inspection reveals that factor as present (since it came from a real reason row).
    """
    pid = str(episode["pid"].iloc[0])
    severe = int(episode["severe_delay"].iloc[0])
    obs = episode[episode["observed"] == 1] if "observed" in episode.columns else episode
    present_factors = sorted({str(f) for f in obs["factor_id"].unique().tolist() if str(f) in all_factors})

    bn = bn.clone_fresh()
    # Clamp never-mentioned factors to absent
    for f in all_factors:
        if f not in present_factors:
            bn.observe(f, False)

    # Only inspect factors that actually appear as reasons for this PID
    remaining = set(present_factors)
    state = PolicyState(bn=bn, remaining=remaining, budget_left=min(budget, len(remaining)))
    inspections: list[str] = []
    steps_to_alert = None

    while state.budget_left > 0 and state.remaining:
        choice = policy.select_next(state)
        if choice is None:
            break
        # Appearing in episode ⇒ present when inspected
        policy.update(state, choice, True)
        inspections.append(choice)
        p = state.bn.query().p_severe
        if steps_to_alert is None and should_alert(p, threshold=threshold):
            steps_to_alert = len(inspections)

    post = state.bn.query()
    alerted = should_alert(post.p_severe, threshold=threshold)
    return Trace(
        pid=pid,
        policy=policy.name,
        budget=budget,
        severe_delay=severe,
        alerted=alerted,
        p_severe=post.p_severe,
        inspections=inspections,
        steps_to_alert=steps_to_alert,
    )
