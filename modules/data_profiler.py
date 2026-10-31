"""
Dataset profiling and analysis for AutoML Model Selector.

Provides detailed statistical summaries, column-type classification,
datetime detection, outlier identification, and correlation analysis.
"""
from __future__ import annotations

import warnings as _warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


# ---------------------------------------------------------------------------
# Column-type helpers
# ---------------------------------------------------------------------------

def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Attempt to parse each object/string column as datetime.

    Args:
        df: Input DataFrame.

    Returns:
        List of column names that could be successfully parsed as datetime.
    """
    datetime_cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue
        if df[col].dtype == object:
            sample = df[col].dropna().head(200)
            if sample.empty:
                continue
            try:
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                ratio = parsed.notna().mean()
                if ratio >= 0.8:
                    datetime_cols.append(col)
            except Exception:
                pass
    return datetime_cols


def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classify each column into a semantic type.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with keys ``'numerical'``, ``'categorical'``,
        ``'datetime'``, ``'text'``, and ``'boolean'``, each mapping to a
        list of column names.
    """
    result: Dict[str, List[str]] = {
        "numerical": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "boolean": [],
    }

    datetime_cols = set(detect_datetime_columns(df))

    for col in df.columns:
        if col in datetime_cols:
            result["datetime"].append(col)
            continue

        series = df[col].dropna()

        if pd.api.types.is_bool_dtype(df[col]):
            result["boolean"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            result["numerical"].append(col)
            continue

        if df[col].dtype == object:
            # Heuristic: treat as "text" if average word count > 5
            try:
                avg_words = series.astype(str).apply(lambda x: len(x.split())).mean()
                if avg_words > 5:
                    result["text"].append(col)
                    continue
            except Exception:
                pass
            result["categorical"].append(col)
            continue

        if hasattr(df[col], "cat"):
            result["categorical"].append(col)
            continue

        # Default fallback
        result["categorical"].append(col)

    return result


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute Pearson correlation matrix for numeric columns.

    Args:
        df: Input DataFrame.

    Returns:
        Correlation matrix as a DataFrame, or ``None`` if fewer than two
        numeric columns exist.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return num_df.corr(method="pearson")


# ---------------------------------------------------------------------------
# Internal profiling helpers
# ---------------------------------------------------------------------------

def _missing_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute missing-value counts and percentages per column."""
    total = len(df)
    result: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        count = int(df[col].isna().sum())
        result[col] = {
            "count": count,
            "percentage": round(100.0 * count / total, 2) if total else 0.0,
        }
    return result


def _numerical_stats(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Compute descriptive statistics for numerical columns."""
    result: Dict[str, Dict[str, Any]] = {}
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            result[col] = {}
            continue
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            try:
                skew = float(_scipy_stats.skew(series))
                kurt = float(_scipy_stats.kurtosis(series))
            except Exception:
                skew = float("nan")
                kurt = float("nan")

        result[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "25%": float(series.quantile(0.25)),
            "50%": float(series.median()),
            "75%": float(series.quantile(0.75)),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
        }
    return result


def _categorical_stats(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Compute frequency statistics for categorical columns."""
    result: Dict[str, Dict[str, Any]] = {}
    for col in cat_cols:
        series = df[col].dropna()
        n_unique = int(series.nunique())
        top = series.value_counts().head(10).to_dict()
        # JSON-serialisable keys
        top = {str(k): int(v) for k, v in top.items()}
        result[col] = {
            "unique_count": n_unique,
            "cardinality_ratio": round(n_unique / len(df), 4) if len(df) else 0.0,
            "top_values": top,
        }
    return result


def _outlier_stats(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Detect outliers using the IQR fence method (1.5 × IQR)."""
    result: Dict[str, Dict[str, Any]] = {}
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            result[col] = {"count": 0, "percentage": 0.0}
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        result[col] = {
            "count": count,
            "percentage": round(100.0 * count / len(series), 2),
            "lower_fence": float(lower),
            "upper_fence": float(upper),
        }
    return result


def _target_analysis(
    df: pd.DataFrame,
    target_col: str,
    col_types: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Analyse the distribution and class balance of the target column.

    Args:
        df:         Input DataFrame.
        target_col: Name of the target column.
        col_types:  Output of :func:`get_column_types`.

    Returns:
        Dictionary with distribution info and, for classification targets,
        an imbalance ratio.
    """
    series = df[target_col].dropna()
    is_categorical = (
        target_col in col_types["categorical"]
        or target_col in col_types["boolean"]
    )

    analysis: Dict[str, Any] = {
        "target_col": target_col,
        "is_categorical": is_categorical,
        "n_missing": int(df[target_col].isna().sum()),
    }

    if is_categorical:
        vc = series.value_counts()
        analysis["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
        analysis["n_classes"] = int(vc.nunique())
        if len(vc) >= 2:
            analysis["imbalance_ratio"] = round(float(vc.iloc[0]) / float(vc.iloc[-1]), 4)
        else:
            analysis["imbalance_ratio"] = 1.0
    else:
        analysis["value_stats"] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }

    return analysis


def _suggested_targets(df: pd.DataFrame, col_types: Dict[str, List[str]]) -> List[str]:
    """Suggest columns that may be reasonable prediction targets.

    Args:
        df:        Input DataFrame.
        col_types: Output of :func:`get_column_types`.

    Returns:
        List of column names ordered by suitability.
    """
    candidates: List[str] = []

    # Binary / low-cardinality categoricals
    for col in col_types["categorical"] + col_types["boolean"]:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 50:
            candidates.append(col)

    # Numerical columns (regression candidates)
    for col in col_types["numerical"]:
        candidates.append(col)

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_dataset(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive statistical profile of a DataFrame.

    Args:
        df:         The DataFrame to profile.
        target_col: Optional name of the prediction target column.

    Returns:
        Dictionary with the following top-level keys:

        * ``shape``            – (rows, cols) tuple
        * ``memory_usage_mb``  – approximate in-memory size in MB
        * ``dtypes``           – per-column dtype strings
        * ``missing``          – per-column missing-value stats
        * ``duplicate_rows``   – count of exact duplicate rows
        * ``column_types``     – output of :func:`get_column_types`
        * ``numerical_stats``  – per-column descriptive statistics
        * ``categorical_stats``– per-column frequency statistics
        * ``outliers``         – per-column IQR outlier counts
        * ``correlations``     – dict form of the Pearson correlation matrix
          (or ``None``)
        * ``target_analysis``  – target-column analysis (or ``None``)
        * ``suggested_targets``– list of candidate target columns
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "Input is not a valid DataFrame."}

    col_types = get_column_types(df)

    corr_matrix = compute_correlations(df)
    corr_dict = None
    if corr_matrix is not None:
        corr_dict = {
            col: {row: round(v, 4) for row, v in corr_matrix[col].items()}
            for col in corr_matrix.columns
        }

    profile: Dict[str, Any] = {
        "shape": df.shape,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": _missing_stats(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_types": col_types,
        "numerical_stats": _numerical_stats(df, col_types["numerical"]),
        "categorical_stats": _categorical_stats(
            df, col_types["categorical"] + col_types["boolean"]
        ),
        "outliers": _outlier_stats(df, col_types["numerical"]),
        "correlations": corr_dict,
        "target_analysis": None,
        "suggested_targets": _suggested_targets(df, col_types),
    }

    if target_col and target_col in df.columns:
        profile["target_analysis"] = _target_analysis(df, target_col, col_types)

    return profile
