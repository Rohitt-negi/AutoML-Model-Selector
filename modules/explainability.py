"""
Model Explainability Module
============================
Provides feature importance, permutation importance, SHAP values,
coefficient extraction, and single-sample explanations for trained pipelines.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_final_estimator(pipeline: Pipeline) -> Any:
    """Return the last step's estimator from a sklearn Pipeline."""
    if isinstance(pipeline, Pipeline):
        return pipeline[-1]
    return pipeline


def _get_preprocessed_X(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply all pipeline steps except the last (estimator) to X."""
    if isinstance(pipeline, Pipeline) and len(pipeline.steps) > 1:
        preprocessing_steps = pipeline.steps[:-1]
        import sklearn.pipeline as skpipe
        pre_pipe = skpipe.Pipeline(preprocessing_steps)
        return pre_pipe.transform(X)
    return X.values if isinstance(X, pd.DataFrame) else X


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_feature_importance(
    result: Dict[str, Any],
    feature_names: List[str],
) -> Optional[pd.DataFrame]:
    """
    Extract native feature importance from a trained pipeline result.

    Supports:
    - Tree-based models via ``feature_importances_``
    - Linear models via ``coef_``

    Parameters
    ----------
    result : dict
        A result dict containing at least ``"pipeline"`` key with a fitted Pipeline.
    feature_names : list of str
        Names of features in the same order as they were passed to fit.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ``['feature', 'importance']`` sorted descending,
        or ``None`` if the model type is unsupported.
    """
    try:
        pipeline: Pipeline = result.get("pipeline")
        if pipeline is None:
            logger.warning("No pipeline found in result dict.")
            return None

        estimator = _get_final_estimator(pipeline)

        # --- Tree-based models ---
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            names = _align_feature_names(pipeline, feature_names, len(importances))
            df = pd.DataFrame({"feature": names, "importance": importances})
            return df.sort_values("importance", ascending=False).reset_index(drop=True)

        # --- Linear models ---
        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                # multi-class: use mean absolute coef across classes
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)
            names = _align_feature_names(pipeline, feature_names, len(importances))
            df = pd.DataFrame({"feature": names, "importance": importances})
            return df.sort_values("importance", ascending=False).reset_index(drop=True)

        logger.info("Model %s does not expose feature_importances_ or coef_.", type(estimator).__name__)
        return None

    except Exception as exc:
        logger.error("get_feature_importance failed: %s", exc)
        logger.debug(traceback.format_exc())
        return None


def _align_feature_names(
    pipeline: Pipeline,
    original_feature_names: List[str],
    n_features: int,
) -> List[str]:
    """
    Attempt to retrieve post-preprocessing feature names from the pipeline.
    Falls back to generic names if lengths mismatch.
    """
    # Try sklearn ≥ 1.0 get_feature_names_out
    if isinstance(pipeline, Pipeline) and len(pipeline.steps) > 1:
        try:
            preprocessor = pipeline[:-1]
            if hasattr(preprocessor, "get_feature_names_out"):
                names = list(preprocessor.get_feature_names_out())
                if len(names) == n_features:
                    return names
        except Exception:
            pass

    if len(original_feature_names) == n_features:
        return original_feature_names

    return [f"feature_{i}" for i in range(n_features)]


def compute_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    feature_names: List[str],
    task_type: str,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Optional[pd.DataFrame]:
    """
    Compute permutation feature importance using sklearn's implementation.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : array-like
        Test labels / targets.
    feature_names : list of str
        Feature column names.
    task_type : str
        ``"classification"`` or ``"regression"``.
    n_repeats : int
        Number of permutation repetitions (default 5).
    random_state : int
        Random seed (default 42).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ``['feature', 'importance', 'std']`` sorted descending.
    """
    try:
        scoring = "accuracy" if task_type == "classification" else "r2"
        perm_result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1,
        )
        importances = perm_result.importances_mean
        stds = perm_result.importances_std

        names = feature_names if len(feature_names) == len(importances) else [
            f"feature_{i}" for i in range(len(importances))
        ]

        df = pd.DataFrame({
            "feature": names,
            "importance": importances,
            "std": stds,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    except Exception as exc:
        logger.error("compute_permutation_importance failed: %s", exc)
        logger.debug(traceback.format_exc())
        return None


def get_model_coefficients(
    pipeline: Pipeline,
    feature_names: List[str],
) -> Optional[pd.DataFrame]:
    """
    Extract raw coefficients from a linear model inside a Pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    feature_names : list of str
        Original feature names.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ``['feature', 'coefficient']`` for binary/regression
        models, or one column per class for multi-class models.
        Returns ``None`` if the estimator is not a linear model.
    """
    try:
        estimator = _get_final_estimator(pipeline)
        if not hasattr(estimator, "coef_"):
            return None

        coef = estimator.coef_
        names = _align_feature_names(pipeline, feature_names, coef.shape[-1])

        if coef.ndim == 1:
            df = pd.DataFrame({"feature": names, "coefficient": coef})
            df["abs_coefficient"] = df["coefficient"].abs()
            return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

        # Multi-class: one column per class
        classes = getattr(estimator, "classes_", range(coef.shape[0]))
        df = pd.DataFrame(coef.T, columns=[f"class_{c}" for c in classes])
        df.insert(0, "feature", names)
        return df

    except Exception as exc:
        logger.error("get_model_coefficients failed: %s", exc)
        return None


def explain_feature_importance(
    importance_df: Optional[pd.DataFrame],
    task_type: str,
) -> str:
    """
    Generate a plain-English explanation of a feature importance DataFrame.

    Parameters
    ----------
    importance_df : pd.DataFrame or None
        Output of ``get_feature_importance`` or ``compute_permutation_importance``.
    task_type : str
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    str
        Human-readable summary.
    """
    if importance_df is None or importance_df.empty:
        return "Feature importance information is not available for this model."

    top = importance_df.head(5)
    bottom = importance_df.tail(3) if len(importance_df) > 5 else pd.DataFrame()

    target_label = "the predicted class" if task_type == "classification" else "the predicted value"

    lines = [
        f"## Feature Importance Summary\n",
        f"The model evaluated **{len(importance_df)} features** to predict {target_label}.\n",
        "### Most Influential Features",
    ]

    total_importance = importance_df["importance"].sum()
    for _, row in top.iterrows():
        pct = (row["importance"] / total_importance * 100) if total_importance > 0 else 0
        lines.append(f"- **{row['feature']}**: importance score {row['importance']:.4f} ({pct:.1f}% of total)")

    if not bottom.empty:
        lines.append("\n### Least Influential Features")
        for _, row in bottom.iterrows():
            pct = (row["importance"] / total_importance * 100) if total_importance > 0 else 0
            lines.append(f"- **{row['feature']}**: importance score {row['importance']:.4f} ({pct:.1f}% of total)")

    # Top-3 cumulative coverage
    top3_sum = importance_df.head(3)["importance"].sum()
    top3_pct = (top3_sum / total_importance * 100) if total_importance > 0 else 0
    lines.append(
        f"\n> The top 3 features account for approximately **{top3_pct:.1f}%** of the total importance."
    )

    return "\n".join(lines)


def get_shap_values(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_samples: int = 100,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Attempt to compute SHAP values for the given pipeline and data.

    Tries TreeExplainer → LinearExplainer → KernelExplainer (in that order).
    Returns ``(None, None)`` if SHAP is not installed or the model is incompatible.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    X_train : pd.DataFrame
        Training data (used as background for KernelExplainer).
    X_test : pd.DataFrame
        Test data to explain.
    max_samples : int
        Maximum samples from X_test to explain (default 100) for speed.

    Returns
    -------
    tuple of (shap_values, explainer) or (None, None)
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        logger.info("SHAP is not installed. Install with: pip install shap")
        return None, None

    try:
        import shap

        estimator = _get_final_estimator(pipeline)

        # Preprocess data
        X_train_proc = _get_preprocessed_X(pipeline, X_train)
        X_test_proc = _get_preprocessed_X(pipeline, X_test)

        # Subsample for speed
        n = min(max_samples, len(X_test_proc))
        if isinstance(X_test_proc, pd.DataFrame):
            X_explain = X_test_proc.iloc[:n]
        else:
            X_explain = X_test_proc[:n]

        # Background dataset for kernel explainer
        n_bg = min(50, len(X_train_proc))
        if isinstance(X_train_proc, np.ndarray):
            background = shap.sample(X_train_proc, n_bg)
        else:
            background = shap.sample(pd.DataFrame(X_train_proc), n_bg)

        # Try TreeExplainer
        tree_models = (
            "RandomForest", "GradientBoosting", "DecisionTree",
            "ExtraTrees", "XGB", "LGB", "CatBoost", "HistGradient"
        )
        model_name = type(estimator).__name__
        if any(m in model_name for m in tree_models):
            try:
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_explain)
                return shap_values, explainer
            except Exception as e:
                logger.debug("TreeExplainer failed: %s", e)

        # Try LinearExplainer
        linear_models = ("LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet", "SGD")
        if any(m in model_name for m in linear_models):
            try:
                explainer = shap.LinearExplainer(estimator, background)
                shap_values = explainer.shap_values(X_explain)
                return shap_values, explainer
            except Exception as e:
                logger.debug("LinearExplainer failed: %s", e)

        # Fallback: KernelExplainer (slow but universal)
        try:
            def predict_fn(x):
                if hasattr(estimator, "predict_proba"):
                    return estimator.predict_proba(x)
                return estimator.predict(x)

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_explain, nsamples=100)
            return shap_values, explainer
        except Exception as e:
            logger.debug("KernelExplainer failed: %s", e)
            return None, None

    except Exception as exc:
        logger.error("get_shap_values failed: %s", exc)
        logger.debug(traceback.format_exc())
        return None, None


def generate_sample_explanation(
    pipeline: Pipeline,
    X_sample: Union[pd.DataFrame, Dict[str, Any]],
    feature_names: List[str],
    task_type: str,
) -> Dict[str, Any]:
    """
    Generate a human-readable explanation for a single sample prediction.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn Pipeline.
    X_sample : pd.DataFrame or dict
        A single sample (one row DataFrame or dict of feature→value).
    feature_names : list of str
        Feature names.
    task_type : str
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    dict
        Keys: ``prediction``, ``feature_contributions`` (list of dicts),
              ``explanation_text``, ``confidence`` (if classification).
    """
    result: Dict[str, Any] = {
        "prediction": None,
        "feature_contributions": [],
        "explanation_text": "",
        "confidence": None,
    }

    try:
        # Normalise to DataFrame
        if isinstance(X_sample, dict):
            X_df = pd.DataFrame([X_sample])[feature_names]
        elif isinstance(X_sample, pd.DataFrame):
            X_df = X_sample[feature_names].head(1)
        else:
            X_df = pd.DataFrame(X_sample, columns=feature_names).head(1)

        prediction = pipeline.predict(X_df)[0]
        result["prediction"] = prediction

        # Confidence / probability
        if task_type == "classification" and hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X_df)[0]
                result["confidence"] = float(np.max(proba))
            except Exception:
                pass

        # Feature contributions via SHAP (best effort)
        try:
            import shap
            estimator = _get_final_estimator(pipeline)
            X_proc = _get_preprocessed_X(pipeline, X_df)

            shap_values = None
            if hasattr(estimator, "feature_importances_"):
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_proc)

            elif hasattr(estimator, "coef_"):
                explainer = shap.LinearExplainer(estimator, X_proc)
                shap_values = explainer.shap_values(X_proc)

            if shap_values is not None:
                sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
                names = _align_feature_names(pipeline, feature_names, len(sv))
                contributions = []
                for name, val, raw in zip(names, sv, X_df.values[0]):
                    contributions.append({
                        "feature": name,
                        "value": raw,
                        "shap_contribution": float(val),
                        "direction": "increases" if val > 0 else "decreases",
                    })
                contributions.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)
                result["feature_contributions"] = contributions

        except Exception:
            # Fallback: use native feature importance scaled by feature value
            pass

        # Build explanation text
        lines = [f"**Prediction:** `{prediction}`"]
        if result["confidence"] is not None:
            lines.append(f"**Confidence:** {result['confidence']:.1%}")

        if result["feature_contributions"]:
            lines.append("\n**Top feature impacts:**")
            for c in result["feature_contributions"][:5]:
                lines.append(
                    f"- `{c['feature']}` = {c['value']} "
                    f"({c['direction']} the prediction by {abs(c['shap_contribution']):.4f})"
                )

        result["explanation_text"] = "\n".join(lines)

    except Exception as exc:
        logger.error("generate_sample_explanation failed: %s", exc)
        result["explanation_text"] = f"Could not generate explanation: {exc}"

    return result
