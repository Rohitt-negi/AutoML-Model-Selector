import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from modules.explainability import get_feature_importance, explain_feature_importance

def test_feature_importance_tree():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    result_dict = {
        "model": clf,
        "pipeline": clf
    }

    imp_df = get_feature_importance(result_dict, feature_names=["featA", "featB"])
    assert imp_df is not None
    assert isinstance(imp_df, pd.DataFrame)
    assert "feature" in imp_df.columns
    assert "importance" in imp_df.columns

    explanation = explain_feature_importance(imp_df, task_type="classification")
    assert isinstance(explanation, str)
    assert len(explanation) > 10

