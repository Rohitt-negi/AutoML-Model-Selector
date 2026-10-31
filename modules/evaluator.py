"""
evaluator.py
============
Metrics and evaluation module for the AutoML Model Selector.

Provides functions to compute task-appropriate evaluation metrics, compare
multiple trained models, identify the best model, and format metrics for
display in the UI.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRIMARY_METRICS: dict[str, str] = {
    "classification": "f1",
    "regression": "r2",
    "clustering": "silhouette_score",
    "time_series": "rmse",
    "nlp": "f1",
    "anomaly_detection": "silhouette_score",
}

# Higher-is-better flag for each metric (False means lower is better)
_HIGHER_IS_BETTER: dict[str, bool] = {
    "accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "roc_auc": True,
    "mae": False,
    "mse": False,
    "rmse": False,
    "r2": True,
    "mape": False,
    "silhouette_score": True,
    "davies_bouldin_score": False,
    "calinski_harabasz_score": True,
    "anomaly_ratio": False,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_primary_metric(task_type: str) -> str:
    """Return the primary evaluation metric name for the given task type.

    Parameters
    ----------
    task_type : str
        Task type string.

    Returns
    -------
    str
        Metric name string (e.g. ``'f1'``, ``'r2'``, ``'silhouette_score'``).

    Raises
    ------
    ValueError
        If ``task_type`` is not recognised.
    """
    task_type = task_type.strip().lower()
    if task_type not in _PRIMARY_METRICS:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            f"Supported types: {', '.join(_PRIMARY_METRICS)}"
        )
    return _PRIMARY_METRICS[task_type]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_pred_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute all relevant evaluation metrics for the given task type.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels or values.
    y_pred : np.ndarray
        Model predictions.
    task_type : str
        Task type string.
    y_pred_proba : np.ndarray or None, optional
        Predicted class probabilities (classification only).

    Returns
    -------
    dict[str, Any]
        Dictionary of metric name → value.
    """
    task_type = task_type.strip().lower()
    dispatch = {
        "classification": _classification_metrics,
        "nlp": _classification_metrics,
        "regression": _regression_metrics,
        "time_series": _regression_metrics,
        "clustering": _clustering_metrics,
        "anomaly_detection": _anomaly_detection_metrics,
    }

    if task_type not in dispatch:
        raise ValueError(f"Unsupported task_type '{task_type}'.")

    fn = dispatch[task_type]
    if task_type in ("classification", "nlp"):
        return fn(y_true, y_pred, y_pred_proba=y_pred_proba)

    return fn(y_true, y_pred)


# ---------------------------------------------------------------------------
# Task-specific metric computations
# ---------------------------------------------------------------------------

def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute classification metrics.

    Handles edge cases: single-class predictions, zero-division, missing probabilities.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    is_binary = n_classes == 2
    average = "binary" if is_binary else "weighted"

    metrics: dict[str, Any] = {}

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Precision / Recall / F1 — suppress zero-division warnings and handle gracefully
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            metrics["precision"] = float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            )
        except Exception as exc:
            logger.debug("Precision computation failed: %s", exc)
            metrics["precision"] = float("nan")

        try:
            metrics["recall"] = float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            )
        except Exception as exc:
            logger.debug("Recall computation failed: %s", exc)
            metrics["recall"] = float("nan")

        try:
            metrics["f1"] = float(
                f1_score(y_true, y_pred, average=average, zero_division=0)
            )
        except Exception as exc:
            logger.debug("F1 computation failed: %s", exc)
            metrics["f1"] = float("nan")

    # ROC-AUC (binary only, requires probability scores)
    if is_binary and y_pred_proba is not None:
        try:
            proba_positive = (
                y_pred_proba[:, 1]
                if y_pred_proba.ndim == 2
                else y_pred_proba
            )
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba_positive))
        except Exception as exc:
            logger.debug("ROC-AUC computation failed: %s", exc)
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    # Confusion matrix
    try:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    except Exception:
        metrics["confusion_matrix"] = []

    # Classification report (text)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics["classification_report"] = classification_report(
                y_true, y_pred, zero_division=0
            )
    except Exception:
        metrics["classification_report"] = ""

    return metrics


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression / time-series metrics.

    Handles edge cases: zero targets for MAPE, NaN predictions.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Replace NaN predictions with mean to prevent cascading NaNs
    nan_mask = np.isnan(y_pred)
    if nan_mask.any():
        logger.warning("%d NaN predictions replaced with mean.", nan_mask.sum())
        y_pred[nan_mask] = np.nanmean(y_pred)

    metrics: dict[str, float] = {}

    try:
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    except Exception:
        metrics["mae"] = float("nan")

    try:
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    except Exception:
        metrics["mse"] = float("nan")

    try:
        metrics["rmse"] = float(np.sqrt(metrics.get("mse", float("nan"))))
    except Exception:
        metrics["rmse"] = float("nan")

    try:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    except Exception:
        metrics["r2"] = float("nan")

    # MAPE — guard against zero targets
    try:
        nonzero_mask = y_true != 0
        if nonzero_mask.sum() == 0:
            metrics["mape"] = float("nan")
        else:
            mape = np.mean(
                np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
            ) * 100.0
            metrics["mape"] = float(mape)
    except Exception:
        metrics["mape"] = float("nan")

    return metrics


def _clustering_metrics(
    y_true: np.ndarray,
    X: np.ndarray,
) -> dict[str, float]:
    """Compute clustering evaluation metrics.

    Note: ``y_true`` is treated as cluster labels here; ``X`` is the feature matrix.
    The function signature mirrors the other metric functions for uniformity.

    Parameters
    ----------
    y_true : np.ndarray
        Cluster label assignments.
    X : np.ndarray
        Feature matrix used to compute distance-based metrics.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    labels = np.asarray(y_true)
    X = np.asarray(X)

    metrics: dict[str, float] = {}

    n_unique_labels = len(set(labels)) - (1 if -1 in labels else 0)

    if n_unique_labels < 2:
        logger.warning(
            "Clustering metrics require at least 2 clusters. "
            "Got %d valid clusters — returning NaN metrics.",
            n_unique_labels,
        )
        return {
            "silhouette_score": float("nan"),
            "davies_bouldin_score": float("nan"),
            "calinski_harabasz_score": float("nan"),
        }

    # Silhouette
    try:
        # Subsample for large datasets to avoid O(n²) computation
        sample_size = min(len(labels), 5000)
        idx = np.random.choice(len(labels), sample_size, replace=False)
        metrics["silhouette_score"] = float(
            silhouette_score(X[idx], labels[idx], sample_size=None)
        )
    except Exception as exc:
        logger.debug("Silhouette score failed: %s", exc)
        metrics["silhouette_score"] = float("nan")

    # Davies-Bouldin
    try:
        # Filter out noise points (label == -1, e.g. DBSCAN)
        mask = labels != -1
        if mask.sum() >= 2 and len(set(labels[mask])) >= 2:
            metrics["davies_bouldin_score"] = float(
                davies_bouldin_score(X[mask], labels[mask])
            )
        else:
            metrics["davies_bouldin_score"] = float("nan")
    except Exception as exc:
        logger.debug("Davies-Bouldin score failed: %s", exc)
        metrics["davies_bouldin_score"] = float("nan")

    # Calinski-Harabasz
    try:
        mask = labels != -1
        if mask.sum() >= 2 and len(set(labels[mask])) >= 2:
            metrics["calinski_harabasz_score"] = float(
                calinski_harabasz_score(X[mask], labels[mask])
            )
        else:
            metrics["calinski_harabasz_score"] = float("nan")
    except Exception as exc:
        logger.debug("Calinski-Harabasz score failed: %s", exc)
        metrics["calinski_harabasz_score"] = float("nan")

    return metrics


def _anomaly_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute anomaly detection metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (1 = normal, -1 = anomaly for sklearn convention; 
        or binary 0/1).
    y_pred : np.ndarray
        Model predictions (same convention as y_true).
    """
    y_pred = np.asarray(y_pred)
    metrics: dict[str, float] = {}

    # Anomaly ratio in predictions
    try:
        anomaly_mask = y_pred == -1
        metrics["anomaly_ratio"] = float(anomaly_mask.mean())
    except Exception:
        metrics["anomaly_ratio"] = float("nan")

    # If ground-truth labels are available and non-trivial, compute supervised metrics
    y_true_arr = np.asarray(y_true)
    unique_true = np.unique(y_true_arr[~np.isnan(y_true_arr.astype(float))])

    if len(unique_true) == 2:
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics["precision"] = float(
                    precision_score(y_true_arr, y_pred, zero_division=0, pos_label=-1)
                )
                metrics["recall"] = float(
                    recall_score(y_true_arr, y_pred, zero_division=0, pos_label=-1)
                )
                metrics["f1"] = float(
                    f1_score(y_true_arr, y_pred, zero_division=0, pos_label=-1)
                )
        except Exception as exc:
            logger.debug("Supervised anomaly metrics failed: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# Model comparison utilities
# ---------------------------------------------------------------------------

def compare_models(
    results_list: list[dict[str, Any]],
    task_type: str,
) -> pd.DataFrame:
    """Build a sorted comparison DataFrame from a list of evaluation results.

    Parameters
    ----------
    results_list : list[dict]
        List of result dicts as returned by the trainer's evaluation functions.
    task_type : str
        Task type string (used to determine sort order).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per model, sorted by the primary metric.
        Columns include ``model_id``, ``model_name``, ``training_time``, and
        all metric columns.
    """
    if not results_list:
        return pd.DataFrame()

    primary_metric = get_primary_metric(task_type)
    ascending = not _HIGHER_IS_BETTER.get(primary_metric, True)

    rows: list[dict[str, Any]] = []
    for result in results_list:
        row: dict[str, Any] = {
            "model_id": result.get("model_id", "unknown"),
            "model_name": result.get("model_name", result.get("model_id", "unknown")),
            "training_time_s": round(result.get("training_time", 0.0), 4),
        }

        # Flatten metrics into top-level columns (skip non-scalar)
        for k, v in result.get("metrics", {}).items():
            if isinstance(v, (int, float)):
                row[k] = round(float(v), 4) if not np.isnan(float(v)) else float("nan")
            elif k == "cv_mean" and isinstance(v, float):
                row[k] = round(v, 4)

        # Add CV scores if present
        cv = result.get("cv_scores")
        if cv and isinstance(cv, dict) and "mean" in cv:
            row["cv_mean"] = round(cv["mean"], 4)
            row["cv_std"] = round(cv["std"], 4)

        if result.get("error"):
            row["error"] = result["error"]

        rows.append(row)

    df = pd.DataFrame(rows)

    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=ascending, na_position="last")

    df = df.reset_index(drop=True)
    return df


def get_best_model(
    results_list: list[dict[str, Any]],
    task_type: str,
) -> dict[str, Any] | None:
    """Return the result dict for the best performing model.

    The best model is determined by the primary metric for the task type.
    Models with errors or NaN primary metrics are excluded.

    Parameters
    ----------
    results_list : list[dict]
        List of result dicts as returned by the trainer's evaluation functions.
    task_type : str
        Task type string.

    Returns
    -------
    dict or None
        The result dict of the best model, or ``None`` if no valid results exist.
    """
    primary_metric = get_primary_metric(task_type)
    higher_is_better = _HIGHER_IS_BETTER.get(primary_metric, True)

    valid_results = [
        r for r in results_list
        if not r.get("error")
        and r.get("metrics")
        and primary_metric in r["metrics"]
        and not _is_nan(r["metrics"][primary_metric])
    ]

    if not valid_results:
        logger.warning("No valid results to select best model from.")
        return None

    return max(valid_results, key=lambda r: _metric_key(r["metrics"][primary_metric], higher_is_better))


def _metric_key(value: float, higher_is_better: bool) -> float:
    """Return a sort key for a metric value."""
    if _is_nan(value):
        return float("-inf") if higher_is_better else float("inf")
    return float(value) if higher_is_better else -float(value)


def _is_nan(value: Any) -> bool:
    """Check whether a value is NaN."""
    try:
        return np.isnan(float(value))
    except (TypeError, ValueError):
        return False


def format_metrics_display(
    metrics: dict[str, Any],
    task_type: str,
) -> dict[str, str]:
    """Return a display-friendly version of the metrics dictionary.

    - Scalar floats are rounded and formatted as percentages where appropriate.
    - Non-scalar values (e.g. confusion matrix, classification report) are kept as-is.
    - NaN values are replaced with the string ``'N/A'``.

    Parameters
    ----------
    metrics : dict
        Raw metrics dict as returned by :func:`compute_metrics`.
    task_type : str
        Task type string.

    Returns
    -------
    dict[str, str]
        Dictionary of metric name → formatted string value.
    """
    percentage_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc", "mape"}

    formatted: dict[str, str] = {}

    for key, value in metrics.items():
        if isinstance(value, float):
            if _is_nan(value):
                formatted[key] = "N/A"
            elif key in percentage_metrics:
                formatted[key] = f"{value * 100:.2f}%" if key != "mape" else f"{value:.2f}%"
            else:
                formatted[key] = f"{value:.4f}"
        elif isinstance(value, int):
            formatted[key] = str(value)
        elif isinstance(value, list):
            # Confusion matrix or similar
            formatted[key] = str(value)
        elif isinstance(value, str):
            formatted[key] = value
        else:
            formatted[key] = str(value)

    return formatted
