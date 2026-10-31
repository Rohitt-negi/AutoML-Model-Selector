import pytest
import pandas as pd
import numpy as np
from modules.data_profiler import profile_dataset, get_column_types, compute_correlations

def test_profile_dataset():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
        "cat1": ["a", "b", "a", "b", "a", "b"],
        "with_null": [1, None, 3, 4, None, 6]
    })

    profile = profile_dataset(df, target_col="cat1")

    assert profile["n_rows"] == 6
    assert profile["n_cols"] == 3
    assert profile["missing_cells"] == 2
    assert profile["total_missing_pct"] > 0
    assert "num1" in profile["numerical_stats"]
    assert "cat1" in profile["categorical_stats"]
    assert profile["target_analysis"] is not None
    assert profile["target_analysis"]["task_family"] == "categorical"


def test_get_column_types():
    df = pd.DataFrame({
        "float_col": [1.1, 2.2, 3.3],
        "int_col": [1, 2, 3],
        "str_col": ["alpha", "beta", "gamma"],
        "bool_col": [True, False, True]
    })

    col_types = get_column_types(df)
    assert "float_col" in col_types["numerical"]
    assert "int_col" in col_types["numerical"]
    assert "str_col" in col_types["categorical"]
    assert "bool_col" in col_types["boolean"]


def test_compute_correlations():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10],
        "text": ["a", "b", "c", "d", "e"]
    })

    corr = compute_correlations(df)
    assert corr is not None
    assert corr.loc["x", "y"] == pytest.approx(1.0, abs=1e-4)

