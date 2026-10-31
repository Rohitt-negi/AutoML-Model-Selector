"""
Prediction Module
==================
Provides single-sample and batch prediction utilities for trained pipelines,
including input validation and human-readable output formatting.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_input(
    input_data: Dict[str, Any],
    feature_names: List[str],
    df_sample: pd.DataFrame,
    task_type: str,
) -> Tuple[bool, List[str]]:
    """
    Validate a user-supplied input dict against the expected features.

    Parameters
    ----------
    input_data : dict
        Mapping of feature_name → value supplied by the user.
    feature_names : list of str
        Expected feature columns.
    df_sample : pd.DataFrame
        A representative sample of the training data (used to check dtypes).
    task_type : str
        ``"classification"`` or ``"regression"`` (informational only here).

    Returns
    -------
    (is_valid, error_messages)
        ``is_valid`` is ``True`` if all checks pass, ``False`` otherwise.
        ``error_messages`` is a list of human-readable error strings.
    """
    errors: List[str] = []

    if not isinstance(input_data, dict):
        return False, ["input_data must be a dictionary of feature_name → value."]

    # 1. Missing required features
    missing = [f for f in feature_names if f not in input_data]
    if missing:
        errors.append(f"Missing required features: {missing}")

    # 2. Unexpected extra features (warn, not error)
    extra = [k for k in input_data if k not in feature_names]
    if extra:
        errors.append(
            f"Unexpected features will be ignored: {extra}. "
            f"Expected: {feature_names}"
        )

    # 3. Type checking for numeric columns
    for col in feature_names:
        if col not in input_data:
            continue  # already captured above
        value = input_data[col]
        if col in df_sample.columns:
            expected_dtype = df_sample[col].dtype
            if pd.api.types.is_numeric_dtype(expected_dtype):
                if not isinstance(value, (int, float, np.integer, np.floating)):
                    try:
                        float(value)  # allow string-encoded numbers
                    except (ValueError, TypeError):
                        errors.append(
                            f"Feature '{col}' expects a numeric value, got: {type(value).__name__} = {value!r}"
                        )
            elif pd.api.types.is_bool_dtype(expected_dtype):
                if not isinstance(value, (bool, np.bool_)):
                    errors.append(
                        f"Feature '{col}' expects a boolean value, got: {type(value).__name__} = {value!r}"
                    )

    # 4. None / NaN values (warn rather than error – imputation handles them)
    null_cols = [
        col for col in feature_names
        if col in input_data and (input_data[col] is None or _is_nan(input_data[col]))
    ]
    if null_cols:
        errors.append(
            f"Features with null values (will be imputed): {null_cols}"
        )

    is_valid = not any(
        "Missing required" in e or "expects a numeric" in e or "expects a boolean" in e
        for e in errors
    )
    return is_valid, errors


def _is_nan(val: Any) -> bool:
    try:
        return bool(np.isnan(val))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Imputation helpers
# ---------------------------------------------------------------------------

def _impute_missing(
    input_data: Dict[str, Any],
    feature_names: List[str],
    df_sample: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Fill missing / None / NaN values using median (numeric) or mode (categorical)
    computed from df_sample.

    Returns a new dict with imputed values.
    """
    filled = {k: v for k, v in input_data.items()}

    for col in feature_names:
        val = filled.get(col)
        if val is None or _is_nan(val):
            if col in df_sample.columns:
                series = df_sample[col].dropna()
                if pd.api.types.is_numeric_dtype(series):
                    filled[col] = float(series.median()) if not series.empty else 0.0
                else:
                    mode_vals = series.mode()
                    filled[col] = mode_vals.iloc[0] if not mode_vals.empty else ""
            else:
                filled[col] = 0  # safe default

    return filled


# ---------------------------------------------------------------------------
# Core prediction functions
# ---------------------------------------------------------------------------

def predict_single(
    pipeline: Pipeline,
    input_data: Dict[str, Any],
    task_type: str,
    feature_names: List[str],
    label_encoder=None,
) -> Dict[str, Any]:
    """
    Run inference for a single sample provided as a feature dict.

    The input goes through the same Pipeline (preprocessing + estimator)
    as the training data.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    input_data : dict
        Feature name → value mapping.
    task_type : str
        ``"classification"``, ``"regression"``, or ``"clustering"``.
    feature_names : list of str
        Ordered list of expected feature names.
    label_encoder : LabelEncoder or None
        If provided, used to decode integer class predictions to original labels.

    Returns
    -------
    dict
        Keys: ``prediction``, ``probabilities`` (classification only),
              ``confidence``, ``raw_prediction``.
    """
    result: Dict[str, Any] = {
        "prediction": None,
        "probabilities": None,
        "confidence": None,
        "raw_prediction": None,
        "error": None,
    }

    try:
        # Build aligned DataFrame with only expected features, in correct order
        aligned = {col: input_data.get(col) for col in feature_names}
        X = pd.DataFrame([aligned], columns=feature_names)

        raw_pred = pipeline.predict(X)[0]
        result["raw_prediction"] = raw_pred

        if task_type == "classification":
            # Decode label if encoder provided
            if label_encoder is not None:
                try:
                    decoded = label_encoder.inverse_transform([int(raw_pred)])[0]
                    result["prediction"] = decoded
                except Exception:
                    result["prediction"] = raw_pred
            else:
                result["prediction"] = raw_pred

            # Probabilities
            if hasattr(pipeline, "predict_proba"):
                try:
                    proba = pipeline.predict_proba(X)[0]
                    classes = pipeline.classes_ if hasattr(pipeline, "classes_") else list(range(len(proba)))
                    if label_encoder is not None:
                        try:
                            classes = label_encoder.inverse_transform(classes)
                        except Exception:
                            pass
                    result["probabilities"] = {str(cls): float(p) for cls, p in zip(classes, proba)}
                    result["confidence"] = float(np.max(proba))
                except Exception as e:
                    logger.debug("predict_proba failed: %s", e)

            elif hasattr(pipeline, "decision_function"):
                try:
                    decision = pipeline.decision_function(X)[0]
                    # Convert to pseudo-probability using sigmoid
                    if np.isscalar(decision):
                        confidence = 1 / (1 + np.exp(-abs(decision)))
                    else:
                        confidence = float(np.max(np.abs(decision)))
                    result["confidence"] = float(confidence)
                except Exception:
                    pass

        elif task_type == "regression":
            result["prediction"] = float(raw_pred)
            result["confidence"] = None  # N/A for regression

        else:  # clustering or other
            result["prediction"] = int(raw_pred) if not isinstance(raw_pred, str) else raw_pred

    except Exception as exc:
        logger.error("predict_single failed: %s", exc)
        logger.debug(traceback.format_exc())
        result["error"] = str(exc)

    return result


def predict_batch(
    pipeline: Pipeline,
    df: pd.DataFrame,
    task_type: str,
    label_encoder=None,
) -> pd.DataFrame:
    """
    Run batch inference on a DataFrame.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    df : pd.DataFrame
        Input features (must contain all columns the pipeline expects).
    task_type : str
        ``"classification"``, ``"regression"``, or ``"clustering"``.
    label_encoder : LabelEncoder or None
        Decodes integer class labels if provided.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with appended columns:
        - ``prediction`` (always)
        - ``confidence`` (classification only)
        - One column per class probability ``prob_<class>`` (classification + predict_proba)
    """
    out_df = df.copy()

    try:
        raw_preds = pipeline.predict(df)

        if task_type == "classification":
            if label_encoder is not None:
                try:
                    decoded = label_encoder.inverse_transform(raw_preds.astype(int))
                    out_df["prediction"] = decoded
                except Exception:
                    out_df["prediction"] = raw_preds
            else:
                out_df["prediction"] = raw_preds

            if hasattr(pipeline, "predict_proba"):
                try:
                    proba = pipeline.predict_proba(df)
                    classes = pipeline.classes_ if hasattr(pipeline, "classes_") else list(range(proba.shape[1]))
                    if label_encoder is not None:
                        try:
                            classes = label_encoder.inverse_transform(classes)
                        except Exception:
                            pass
                    for i, cls in enumerate(classes):
                        out_df[f"prob_{cls}"] = proba[:, i]
                    out_df["confidence"] = np.max(proba, axis=1)
                except Exception as e:
                    logger.debug("predict_proba in batch failed: %s", e)

        elif task_type == "regression":
            out_df["prediction"] = raw_preds.astype(float)

        else:
            out_df["prediction"] = raw_preds

    except Exception as exc:
        logger.error("predict_batch failed: %s", exc)
        logger.debug(traceback.format_exc())
        out_df["prediction"] = None
        out_df["error"] = str(exc)

    return out_df


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_prediction_output(
    prediction_dict: Dict[str, Any],
    task_type: str,
    label_encoder=None,
) -> Dict[str, Any]:
    """
    Convert a raw prediction dict (from ``predict_single``) to a
    human-readable, display-friendly format.

    Parameters
    ----------
    prediction_dict : dict
        Output of ``predict_single``.
    task_type : str
        ``"classification"`` or ``"regression"``.
    label_encoder : LabelEncoder or None

    Returns
    -------
    dict
        Keys: ``result_summary``, ``display_prediction``, ``detail_lines``,
              ``probabilities_table`` (classification only).
    """
    out: Dict[str, Any] = {
        "prediction": prediction_dict.get("prediction"),
        "probabilities": prediction_dict.get("probabilities"),
        "result_summary": "",
        "display_prediction": "",
        "detail_lines": [],
        "probabilities_table": None,
    }

    if prediction_dict.get("error"):
        out["result_summary"] = f"⚠️ Prediction failed: {prediction_dict['error']}"
        return out

    pred = prediction_dict.get("prediction")
    confidence = prediction_dict.get("confidence")
    proba = prediction_dict.get("probabilities")

    if task_type == "classification":
        out["display_prediction"] = str(pred)
        detail = [f"**Predicted class:** `{pred}`"]
        if confidence is not None:
            detail.append(f"**Confidence:** {confidence:.1%}")
        out["detail_lines"] = detail

        if proba:
            rows = []
            for cls, p in sorted(proba.items(), key=lambda x: x[1], reverse=True):
                rows.append({"Class": cls, "Probability": f"{p:.4f}", "Percentage": f"{p*100:.1f}%"})
            out["probabilities_table"] = pd.DataFrame(rows)

        if confidence is not None:
            if confidence >= 0.90:
                level = "🟢 High"
            elif confidence >= 0.70:
                level = "🟡 Medium"
            else:
                level = "🔴 Low"
            out["result_summary"] = (
                f"Predicted **{pred}** with {confidence:.1%} confidence ({level} confidence)."
            )
        else:
            out["result_summary"] = f"Predicted class: **{pred}**"

    elif task_type == "regression":
        pred_val = float(pred) if pred is not None else None
        out["display_prediction"] = f"{pred_val:.4f}" if pred_val is not None else "N/A"
        out["result_summary"] = f"Predicted value: **{pred_val:.4f}**" if pred_val is not None else "N/A"
        out["detail_lines"] = [f"**Predicted value:** `{pred_val:.4f}`"]

    else:
        out["display_prediction"] = str(pred)
        out["result_summary"] = f"Cluster assignment: **{pred}**"
        out["detail_lines"] = [f"**Assigned cluster:** `{pred}`"]

    return out
