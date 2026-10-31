import pytest
import numpy as np
import pandas as pd
from modules.evaluator import compute_metrics, compare_models, get_best_model, get_primary_metric

def test_compute_classification_metrics():
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0])

    metrics = compute_metrics(y_true, y_pred, task_type="classification")
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_compute_regression_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 19.0, 31.0, 38.0])

    metrics = compute_metrics(y_true, y_pred, task_type="regression")
    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] >= 0.0


def test_compare_and_get_best_model():
    results = [
        {
            "model_id": "model_a",
            "model_name": "Model A",
            "metrics": {"accuracy": 0.85, "f1": 0.84, "precision": 0.85, "recall": 0.83},
            "training_time": 0.12
        },
        {
            "model_id": "model_b",
            "model_name": "Model B",
            "metrics": {"accuracy": 0.92, "f1": 0.91, "precision": 0.92, "recall": 0.90},
            "training_time": 0.45
        }
    ]

    comp_df = compare_models(results, task_type="classification")
    assert isinstance(comp_df, pd.DataFrame)
    assert len(comp_df) == 2

    best = get_best_model(results, task_type="classification")
    assert best["model_id"] == "model_b"

