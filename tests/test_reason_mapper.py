from src.data.reason_mapper import map_reason_regex


def test_budget_reason():
    assert map_reason_regex("PROJECT DELAYED DUE TO BUDGETARY CONSTRAINTS") == "budget"


def test_scope_reason():
    assert map_reason_regex("PROJECT DELAYED DUE TO CHANGES IN SCOPE/DESIGN") == "scope_design"


def test_fallback():
    assert map_reason_regex("COMPLETELY UNKNOWN REASON XYZ") == "other_admin"
