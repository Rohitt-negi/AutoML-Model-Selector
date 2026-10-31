"""
Automatic ML problem-type detection for AutoML Model Selector.

Given a DataFrame and an optional target column, this module infers whether
the problem is classification, regression, clustering, time-series, NLP, or
anomaly detection, together with sub-type, confidence score, and reasoning.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Task type descriptions
# ---------------------------------------------------------------------------

_TASK_DESCRIPTIONS: Dict[str, str] = {
    "classification": (
        "Classification: predict which discrete category or class an instance "
        "belongs to. Typical algorithms include logistic regression, decision "
        "trees, random forests, gradient boosting, and neural networks."
    ),
    "regression": (
        "Regression: predict a continuous numerical value. Typical algorithms "
        "include linear regression, ridge/lasso, decision trees, gradient "
        "boosting, and neural networks."
    ),
    "clustering": (
        "Clustering: discover natural groupings in unlabelled data. Typical "
        "algorithms include K-Means, DBSCAN, hierarchical clustering, and "
        "Gaussian mixture models."
    ),
    "time_series": (
        "Time-Series Forecasting: predict future values based on historical "
        "sequential observations ordered by time. Typical algorithms include "
        "ARIMA, Prophet, LSTM, and gradient-boosting regressors with lag "
        "features."
    ),
    "nlp": (
        "Natural Language Processing (NLP): extract meaning or make "
        "predictions from free-form text data. Typical approaches include "
        "TF-IDF with classical classifiers, transformer-based models (BERT, "
        "GPT), and sequence-to-sequence architectures."
    ),
    "anomaly_detection": (
        "Anomaly Detection: identify rare or unusual instances that deviate "
        "significantly from the majority of the data. Typical algorithms "
        "include Isolation Forest, One-Class SVM, Local Outlier Factor, and "
        "autoencoders."
    ),
}


def get_task_description(task_type: str) -> str:
    """Return a human-readable description of the given task type.

    Args:
        task_type: One of ``'classification'``, ``'regression'``,
            ``'clustering'``, ``'time_series'``, ``'nlp'``,
            ``'anomaly_detection'``.

    Returns:
        Description string, or a generic fallback if the type is unknown.
    """
    return _TASK_DESCRIPTIONS.get(
        task_type,
        f"Unknown task type: '{task_type}'. "
        "Please consult the documentation for supported task types.",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_datetime_col(series: pd.Series) -> bool:
    """Return True if *series* looks like a datetime column."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        sample = series.dropna().head(200).astype(str)
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            return parsed.notna().mean() >= 0.8
        except Exception:
            pass
    return False


def _has_sequential_datetime(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame has at least one sequential datetime column."""
    for col in df.columns:
        if _is_datetime_col(df[col]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                non_null = parsed.dropna()
                if len(non_null) < 10:
                    continue
                diffs = non_null.sort_values().diff().dropna()
                # A sequential index has mostly uniform positive diffs.
                if (diffs > pd.Timedelta(0)).mean() >= 0.8:
                    return True
            except Exception:
                pass
    return False


def _has_text_columns(df: pd.DataFrame, avg_word_threshold: float = 5.0) -> bool:
    """Return True if at least one object column looks like free-form text."""
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(500).astype(str)
            try:
                avg_words = sample.apply(lambda x: len(x.split())).mean()
                if avg_words > avg_word_threshold:
                    return True
            except Exception:
                pass
    return False


def _is_integer_like(series: pd.Series) -> bool:
    """Return True if a numeric series contains only integer values."""
    cleaned = series.dropna()
    if cleaned.empty:
        return False
    return bool((cleaned == cleaned.round()).all())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_task_type(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    user_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Infer the ML task type from the dataset and target column.

    Args:
        df:         The input DataFrame.
        target_col: Name of the column to be predicted. If ``None``, an
                    unsupervised task (clustering or anomaly detection) is
                    inferred.
        user_hint:  Optional free-text hint from the user (e.g.
                    ``'classification'``, ``'regression'``). When provided
                    and valid, it overrides the automatic inference.

    Returns:
        Dictionary with the following keys:

        * ``'task_type'``   – One of ``'classification'``, ``'regression'``,
          ``'clustering'``, ``'time_series'``, ``'nlp'``,
          ``'anomaly_detection'``.
        * ``'sub_type'``    – More specific label (e.g.
          ``'binary_classification'``, ``'multivariate_regression'``).
        * ``'confidence'``  – Float in [0, 1] reflecting detection certainty.
        * ``'reasoning'``   – List of strings explaining the inference steps.
        * ``'warnings'``    – List of potential issue strings.
        * ``'description'`` – Human-readable task description.
    """
    reasoning: List[str] = []
    warnings_out: List[str] = []
    confidence = 0.5
    task_type = "clustering"
    sub_type = "unsupervised_clustering"

    # ------------------------------------------------------------------
    # 0. Apply user hint if valid
    # ------------------------------------------------------------------
    valid_hints = set(_TASK_DESCRIPTIONS.keys())
    if user_hint and user_hint.lower() in valid_hints:
        task_type = user_hint.lower()
        reasoning.append(f"User explicitly specified task type: '{task_type}'.")
        confidence = 0.95
        sub_type = _derive_sub_type(task_type, df, target_col, reasoning, warnings_out)
        return _build_result(task_type, sub_type, confidence, reasoning, warnings_out)

    # ------------------------------------------------------------------
    # 1. No target column → unsupervised
    # ------------------------------------------------------------------
    if not target_col or target_col not in df.columns:
        reasoning.append("No target column specified → unsupervised task.")
        if _has_sequential_datetime(df):
            task_type = "time_series"
            sub_type = "unsupervised_time_series"
            confidence = 0.6
            reasoning.append("Sequential datetime column detected; may be a time-series dataset.")
        elif df.shape[0] > 0 and (df.shape[0] / max(df.shape[1], 1)) > 50:
            # Many rows relative to columns → clustering is more useful.
            task_type = "clustering"
            sub_type = "unsupervised_clustering"
            confidence = 0.55
            reasoning.append(
                "High row-to-column ratio suggests clustering is a sensible default."
            )
        else:
            task_type = "anomaly_detection"
            sub_type = "unsupervised_anomaly_detection"
            confidence = 0.5
            reasoning.append(
                "Low row-to-column ratio; anomaly detection may be more appropriate."
            )

        if _has_text_columns(df):
            warnings_out.append(
                "Free-form text columns detected. Consider an NLP approach."
            )
        return _build_result(task_type, sub_type, confidence, reasoning, warnings_out)

    # ------------------------------------------------------------------
    # 2. Target column present
    # ------------------------------------------------------------------
    target = df[target_col]
    reasoning.append(f"Target column '{target_col}' found (dtype: {target.dtype}).")

    # ------------------------------------------------------------------
    # 2a. Time-series check (takes priority)
    # ------------------------------------------------------------------
    if _has_sequential_datetime(df):
        task_type = "time_series"
        reasoning.append(
            "Sequential datetime column detected → time-series forecasting task."
        )
        if pd.api.types.is_numeric_dtype(target):
            sub_type = "univariate_regression" if df.shape[1] <= 3 else "multivariate_regression"
        else:
            sub_type = "time_series_classification"
        confidence = 0.80
        return _build_result(task_type, sub_type, confidence, reasoning, warnings_out)

    # ------------------------------------------------------------------
    # 2b. NLP check (text-heavy feature columns)
    # ------------------------------------------------------------------
    if _has_text_columns(df):
        reasoning.append("Free-form text columns detected.")
        if pd.api.types.is_bool_dtype(target) or (
            target.dtype == object and target.nunique() <= 20
        ):
            task_type = "nlp"
            sub_type = (
                "binary_text_classification"
                if target.nunique() <= 2
                else "multiclass_text_classification"
            )
            confidence = 0.75
            reasoning.append("Categorical target with text features → NLP classification.")
        else:
            task_type = "nlp"
            sub_type = "text_regression"
            confidence = 0.65
            reasoning.append("Numerical target with text features → NLP regression.")
        return _build_result(task_type, sub_type, confidence, reasoning, warnings_out)

    # ------------------------------------------------------------------
    # 2c. Boolean target → binary classification
    # ------------------------------------------------------------------
    if pd.api.types.is_bool_dtype(target):
        task_type = "classification"
        sub_type = "binary_classification"
        confidence = 0.95
        reasoning.append("Boolean target column → binary classification.")
        return _build_result(task_type, sub_type, confidence, reasoning, warnings_out)

    # ------------------------------------------------------------------
    # 2d. Categorical / string target
    # ------------------------------------------------------------------
    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or hasattr(target, "cat")
        or isinstance(target.dtype, pd.CategoricalDtype)
    ):
        n_unique = int(target.nunique())
        reasoning.append(
            f"Target is categorical with {n_unique} unique value(s)."
        )
        if n_unique == 1:
            warnings_out.append(
                "Target has only one unique value; classification is meaningless."
            )
            task_type = "classification"
            sub_type = "degenerate_single_class"
            confidence = 0.2
        elif n_unique == 2:
            task_type = "classification"
            sub_type = "binary_classification"
            confidence = 0.95
            reasoning.append("Exactly 2 classes → binary classification.")
        elif n_unique <= 20:
            task_type = "classification"
            sub_type = "multiclass_classification"
            confidence = 0.90
            reasoning.append(f"{n_unique} classes (≤20) → multiclass classification.")
        else:
            task_type = "classification"
            sub_type = "high_cardinality_classification"
            confidence = 0.55
            warnings_out.append(
                f"Target has {n_unique} unique values (>20). "
                "Consider whether this is truly a classification problem or "
                "if label encoding / target encoding is needed."
            )
            reasoning.append(f"{n_unique} unique target values > 20 — high cardinality warning.")
        return _build_result(task_type, sub_type, confidence, reasoning, warnings_out)

    # ------------------------------------------------------------------
    # 2e. Numerical target → regression (or possible classification)
    # ------------------------------------------------------------------
    if pd.api.types.is_numeric_dtype(target):
        n_unique = int(target.nunique())
        reasoning.append(
            f"Target is numeric with {n_unique} unique value(s)."
        )

        # Could it be integer-encoded classes?
        if _is_integer_like(target) and n_unique <= 20:
            task_type = "classification"
            sub_type = (
                "binary_classification" if n_unique == 2 else "multiclass_classification"
            )
            confidence = 0.72
            reasoning.append(
                f"Numeric target contains only integer values with ≤20 unique values "
                f"→ likely integer-encoded classes (classification)."
            )
            warnings_out.append(
                "Target looks like integer-encoded class labels. "
                "If it represents a continuous measure, switch to regression."
            )
        else:
            task_type = "regression"
            n_cols = df.shape[1] - 1  # exclude target
            sub_type = "univariate_regression" if n_cols <= 1 else "multivariate_regression"
            confidence = 0.88
            reasoning.append("Continuous numeric target → regression task.")

        return _build_result(task_type, sub_type, confidence, reasoning, warnings_out)

    # ------------------------------------------------------------------
    # 2f. Fallback
    # ------------------------------------------------------------------
    warnings_out.append(
        f"Could not confidently determine task type for target dtype '{target.dtype}'. "
        "Defaulting to classification."
    )
    task_type = "classification"
    sub_type = "unknown_classification"
    confidence = 0.3
    return _build_result(task_type, sub_type, confidence, reasoning, warnings_out)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _derive_sub_type(
    task_type: str,
    df: pd.DataFrame,
    target_col: Optional[str],
    reasoning: List[str],
    warnings_out: List[str],
) -> str:
    """Derive a sub-type label when the task type is user-specified.

    Args:
        task_type:   The validated task type string.
        df:          Input DataFrame.
        target_col:  Target column name (may be ``None``).
        reasoning:   Mutable list to append reasoning notes to.
        warnings_out: Mutable list to append warnings to.

    Returns:
        Sub-type string.
    """
    if task_type == "classification" and target_col and target_col in df.columns:
        n_unique = int(df[target_col].nunique())
        if n_unique == 2:
            reasoning.append("Binary target detected.")
            return "binary_classification"
        elif n_unique <= 20:
            return "multiclass_classification"
        else:
            warnings_out.append(f"Target has {n_unique} unique values — high cardinality.")
            return "high_cardinality_classification"

    if task_type == "regression":
        n_cols = df.shape[1] - (1 if target_col else 0)
        return "univariate_regression" if n_cols <= 1 else "multivariate_regression"

    if task_type == "time_series":
        return "univariate_time_series"

    if task_type == "nlp":
        return "text_classification"

    return f"unsupervised_{task_type}"


def _build_result(
    task_type: str,
    sub_type: str,
    confidence: float,
    reasoning: List[str],
    warnings_out: List[str],
) -> Dict[str, Any]:
    """Assemble the final detection result dictionary.

    Args:
        task_type:    Detected task type string.
        sub_type:     Detected sub-type string.
        confidence:   Confidence float in [0, 1].
        reasoning:    List of reasoning strings.
        warnings_out: List of warning strings.

    Returns:
        Fully populated result dictionary.
    """
    return {
        "task_type": task_type,
        "sub_type": sub_type,
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3),
        "reasoning": reasoning,
        "warnings": warnings_out,
        "description": get_task_description(task_type),
    }
