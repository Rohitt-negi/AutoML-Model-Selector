import pytest
import pandas as pd
import numpy as np
from modules.task_detector import detect_task_type, get_task_description

def test_detect_binary_classification():
    df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "target": np.random.choice(["Yes", "No"], size=50)
    })

    result = detect_task_type(df, target_col="target")
    assert result["task_type"] == "classification"
    assert result["sub_type"] == "binary_classification"
    assert result["confidence"] > 0.8


def test_detect_regression():
    df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "target": np.random.uniform(10.0, 100.0, size=50)
    })

    result = detect_task_type(df, target_col="target")
    assert result["task_type"] == "regression"
    assert "regression" in result["sub_type"]


def test_detect_clustering_when_no_target():
    df = pd.DataFrame({
        "f1": np.random.randn(40),
        "f2": np.random.randn(40),
        "f3": np.random.randn(40)
    })

    result = detect_task_type(df, target_col=None)
    assert result["task_type"] in ["clustering", "anomaly_detection"]


def test_task_description():
    desc = get_task_description("classification")
    assert "class" in desc.lower() or "category" in desc.lower()

