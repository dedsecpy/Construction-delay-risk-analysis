from src.bn.inference import DelayRiskBN


def test_prior_between_0_and_1():
    bn = DelayRiskBN()
    post = bn.query()
    assert 0.0 <= post.p_severe <= 1.0


def test_positive_evidence_raises_severe_prob():
    bn = DelayRiskBN(factor_priors={f: 0.1 for f in ["budget", "scope_design", "grants_approvals", "procurement", "construction_conditions", "external_dependency", "workforce_resource", "other_admin"]})
    before = bn.query().p_severe
    for f in bn.factors:
        bn.observe(f, True)
    after = bn.query().p_severe
    assert after > before
