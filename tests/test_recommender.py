import pytest
from modules.recommender import get_model_candidates, explain_recommendation

def test_recommendation_classification():
    profile = {
        "n_rows": 500,
        "n_cols": 10,
        "total_missing_pct": 2.5,
        "column_types": {"numerical": ["a", "b"], "categorical": ["c"]}
    }

    candidates = get_model_candidates("classification", profile, sub_type="binary_classification")
    assert len(candidates) > 0

    # Ensure required metadata keys exist
    for c in candidates:
        assert "name" in c
        assert "id" in c
        assert "category" in c
        assert "why_selected" in c
        assert "advantages" in c
        assert "limitations" in c
        assert "computational_cost" in c
        assert "metrics" in c

    # One model should be marked as recommended
    has_recommended = any(c.get("is_recommended", False) for c in candidates)
    assert has_recommended


def test_recommendation_regression():
    profile = {
        "n_rows": 300,
        "n_cols": 5,
        "total_missing_pct": 0.0
    }

    candidates = get_model_candidates("regression", profile)
    assert len(candidates) > 0
    candidate_ids = [c["id"] for c in candidates]
    assert "linear_regression" in candidate_ids or "ridge" in candidate_ids


def test_explain_recommendation():
    profile = {"n_rows": 250, "n_cols": 8, "total_missing_pct": 1.0}
    candidates = get_model_candidates("classification", profile)
    exp = explain_recommendation(candidates[0], profile)
    assert isinstance(exp, str)
    assert len(exp) > 20

