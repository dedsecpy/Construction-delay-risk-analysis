from src.bn.inference import DelayRiskBN
from src.policy.base import PolicyState
from src.policy.consensus_voi import ConsensusVoIPolicy
from src.policy.fixed_ranking import FixedRankingPolicy


def test_fixed_respects_order():
    factors = ["budget", "scope_design", "procurement"]
    bn = DelayRiskBN(factor_priors={f: 0.2 for f in factors})
    # monkeypatch factors list by using full BN but remaining subset
    state = PolicyState(bn=bn, remaining=set(factors), budget_left=2)
    pol = FixedRankingPolicy(["procurement", "budget", "scope_design"])
    assert pol.select_next(state) == "procurement"


def test_voi_returns_something():
    bn = DelayRiskBN()
    state = PolicyState(bn=bn, remaining=set(bn.factors), budget_left=2)
    pol = ConsensusVoIPolicy(lookahead=1)
    choice = pol.select_next(state)
    assert choice in bn.factors
