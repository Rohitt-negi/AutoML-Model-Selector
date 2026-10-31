import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from modules.predictor import predict_single, validate_input, format_prediction_output

def test_validate_input():
    df_sample = pd.DataFrame({"age": [20, 30, 40], "salary": [2000, 3000, 4000]})
    user_input = {"age": 25, "salary": 2500}

    is_valid, errors = validate_input(user_input, ["age", "salary"], df_sample, "classification")
    assert is_valid
    assert len(errors) == 0


def test_predict_single_and_format():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [5.0, 6.0], [6.0, 7.0]])
    y = np.array([0, 0, 1, 1])

    clf = LogisticRegression()
    clf.fit(X, y)

    res = predict_single(
        pipeline=clf,
        input_data={"f1": 5.5, "f2": 6.5},
        task_type="classification",
        feature_names=["f1", "f2"]
    )

    assert "prediction" in res
    assert res["prediction"] in [0, 1]

    formatted = format_prediction_output(res, task_type="classification")
    assert "prediction" in formatted

