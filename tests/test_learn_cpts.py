import pandas as pd

from src.bn.learn_cpts import learn_noisy_or_params


def test_learn_noisy_or_on_tiny():
    rows = []
    for pid, severe, factors in [
        ("1", 1, ["budget", "scope_design"]),
        ("2", 0, ["other_admin"]),
        ("3", 1, ["budget"]),
        ("4", 0, ["procurement"]),
        ("5", 1, ["budget", "grants_approvals"]),
    ]:
        for i, f in enumerate(factors):
            rows.append(
                {
                    "pid": pid,
                    "step": i,
                    "factor_id": f,
                    "severe_delay": severe,
                    "split": "train",
                    "variance_day": 1.0,
                    "cumulative_variance": float(i + 1),
                    "observed": 1,
                }
            )
    df = pd.DataFrame(rows)
    out = learn_noisy_or_params(df)
    assert 0.05 <= out["p_cause"] <= 0.8
    assert 0.0 < out["leak"] <= 0.2
    assert "consensus_priors" in out
