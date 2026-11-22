from __future__ import annotations

from src.bn.inference import DelayRiskBN
from src.policy.base import Policy, PolicyState
from src.policy.utility import expected_alert_utility


class ConsensusVoIPolicy(Policy):
    name = "consensus_voi"

    def __init__(self, lookahead: int = 1):
        self.lookahead = lookahead

    def select_next(self, state: PolicyState) -> str | None:
        if state.budget_left <= 0 or not state.remaining:
            return None
        return _best_inspection(state, self.lookahead)


def _clone_bn(bn: DelayRiskBN) -> DelayRiskBN:
    twin = bn.clone_fresh()
    twin.evidence = dict(bn.evidence)
    return twin


def _best_inspection(state: PolicyState, lookahead: int) -> str | None:
    best_f = None
    best_score = float("-inf")
    for f in state.remaining:
        score = _expected_utility_after(state, f, lookahead)
        if score > best_score:
            best_score = score
            best_f = f
    return best_f


def _expected_utility_after(state: PolicyState, factor: str, lookahead: int) -> float:
    post = state.bn.query()
    p_present = post.factor_marginals.get(factor, 0.5)
    total = 0.0
    for obs, p_obs in ((1, p_present), (0, 1 - p_present)):
        bn2 = _clone_bn(state.bn)
        bn2.observe(factor, bool(obs))
        if lookahead <= 1 or state.budget_left <= 1:
            total += p_obs * expected_alert_utility(bn2.query().p_severe)
            continue
        rem = set(state.remaining)
        rem.discard(factor)
        st2 = PolicyState(bn=bn2, remaining=rem, budget_left=state.budget_left - 1)
        nxt = _best_inspection(st2, 1)
        if nxt is None:
            total += p_obs * expected_alert_utility(bn2.query().p_severe)
        else:
            total += p_obs * _expected_utility_after(st2, nxt, 1)
    return total
