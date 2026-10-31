import pytest
import pandas as pd
import numpy as np
from modules.model_trainer import evaluate_quick
from modules.data_profiler import profile_dataset

def test_evaluate_quick_classification():
    # Synthetic classification data
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "cat_feature": np.random.choice(["typeA", "typeB"], size=n),
        "target": np.random.choice(["Class1", "Class2"], size=n)
    })

    profile = profile_dataset(df, target_col="target")
    results = evaluate_quick(
        df=df,
        target_col="target",
        task_type="classification",
        profile=profile,
        random_state=42
    )

    assert isinstance(results, list)
    assert len(results) > 0

    first = results[0]
    assert "model_id" in first
    assert "model_name" in first
    assert "metrics" in first
    assert "accuracy" in first["metrics"]
    assert "training_time" in first
    assert first["pipeline"] is not None


def test_evaluate_quick_regression():
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "f1": np.random.uniform(1.0, 10.0, n),
        "f2": np.random.uniform(10.0, 50.0, n),
        "price": np.random.uniform(100.0, 500.0, n)
    })

    profile = profile_dataset(df, target_col="price")
    results = evaluate_quick(
        df=df,
        target_col="price",
        task_type="regression",
        profile=profile,
        random_state=42
    )

    assert len(results) > 0
    assert "rmse" in results[0]["metrics"]
    assert "r2" in results[0]["metrics"]

