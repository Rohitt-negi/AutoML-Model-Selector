import pytest
import pandas as pd
import numpy as np
from modules.preprocessing import (
    build_preprocessing_pipeline,
    split_data,
    encode_target,
    check_data_leakage_risks
)

def test_split_data_and_encode_target():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
    })

    X_train, X_test, y_train, y_test = split_data(df, target_col="target", task_type="classification", test_size=0.2)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert "target" not in X_train.columns

    y_enc, enc = encode_target(y_train, task_type="classification")
    assert enc is not None
    assert set(y_enc).issubset({0, 1})


def test_build_pipeline_transform():
    df = pd.DataFrame({
        "num": [10.0, 20.0, np.nan, 40.0, 50.0],
        "cat": ["Low", "High", "Medium", "Low", np.nan]
    })

    preprocessor = build_preprocessing_pipeline(
        df=df,
        target_col=None,
        task_type="clustering",
        numerical_cols=["num"],
        categorical_cols=["cat"]
    )

    transformed = preprocessor.fit_transform(df)
    assert transformed is not None
    assert not np.isnan(transformed).any()


def test_data_leakage_detection():
    df = pd.DataFrame({
        "id_col": list(range(1, 101)),
        "target": np.random.choice([0, 1], size=100),
        "feat": np.random.randn(100)
    })

    warnings = check_data_leakage_risks(df, target_col="target")
    assert isinstance(warnings, list)

