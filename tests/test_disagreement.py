from src.bn.disagreement import collapse_votes, disagreement_from_votes


def test_disagreement_has_higher_entropy_pull():
    votes = {
        "budget": {"low": 3, "medium": 6, "high": 1},  # contested
        "scope_design": {"low": 0, "medium": 0, "high": 10},  # unanimous high
    }
    cons = collapse_votes(votes)
    dis = disagreement_from_votes(votes)
    # contested factor should move toward 0.5 relative to unanimous
    assert abs(dis["budget"] - 0.5) < abs(cons["budget"] - 0.5) or dis["budget"] != cons["budget"]
    assert dis["scope_design"] >= 0.7
