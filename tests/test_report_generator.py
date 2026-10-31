import pytest
import pandas as pd
from modules.report_generator import generate_dataset_report, generate_model_comparison_csv

def test_generate_dataset_report():
    df = pd.DataFrame({"colA": [1, 2, 3], "colB": ["x", "y", "z"]})
    profile = {
        "n_rows": 3,
        "n_cols": 2,
        "memory_usage": "10 KB",
        "duplicate_rows": 0,
        "total_missing_pct": 0.0,
        "missing_values": {},
        "column_types": {"numerical": ["colA"], "categorical": ["colB"]},
        "numerical_stats": {},
        "categorical_stats": {},
        "outliers": {}
    }

    report = generate_dataset_report(profile, df)
    assert isinstance(report, str)
    assert "# Dataset Analysis Report" in report or "Report" in report


def test_generate_model_comparison_csv():
    comp_df = pd.DataFrame({
        "Model": ["LogReg", "RandomForest"],
        "Accuracy": [0.85, 0.92]
    })

    csv_bytes = generate_model_comparison_csv(comp_df)
    assert isinstance(csv_bytes, bytes)
    assert b"LogReg" in csv_bytes

