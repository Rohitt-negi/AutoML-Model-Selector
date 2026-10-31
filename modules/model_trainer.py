"""
model_trainer.py
================
Model training and evaluation orchestrator for the AutoML Model Selector.

Provides helpers to instantiate sklearn-compatible estimators, train them
through a preprocessing pipeline, and run quick/standard/advanced evaluation
loops with consistent result dictionaries.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from modules.evaluator import compute_metrics, get_primary_metric
from modules.preprocessing import (
    build_preprocessing_pipeline,
    encode_target,
    split_data,
)
from modules.recommender import get_model_candidates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb  # type: ignore

    _XGBOOST_AVAILABLE = True
except Exception:
    _XGBOOST_AVAILABLE = False
    logger.debug("XGBoost not available — xgboost models will be skipped.")

try:
    import lightgbm as lgb  # type: ignore

    _LIGHTGBM_AVAILABLE = True
except Exception:
    _LIGHTGBM_AVAILABLE = False
    logger.debug("LightGBM not available — lightgbm models will be skipped.")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_sklearn_model(
    model_id: str,
    task_type: str,
    random_state: int = 42,
) -> BaseEstimator:
    """Return an unfitted sklearn-compatible estimator for the given model ID.

    Parameters
    ----------
    model_id : str
        Snake-case model identifier as used in :mod:`modules.recommender`.
    task_type : str
        Task type string (used to disambiguate shared IDs).
    random_state : int, optional
        Random seed for reproducible results (default 42).

    Returns
    -------
    BaseEstimator
        An unfitted sklearn estimator.

    Raises
    ------
    ValueError
        If ``model_id`` is not recognised or its dependency is not installed.
    """
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        IsolationForest,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import (
        Lasso,
        LinearRegression,
        LogisticRegression,
        Ridge,
    )
    from sklearn.mixture import GaussianMixture
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
    from sklearn.svm import SVC, SVR, OneClassSVM
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

    registry: dict[str, BaseEstimator] = {
        # --- Classification ---
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "svm": SVC(probability=True, random_state=random_state),
        "knn": KNeighborsClassifier(n_jobs=-1),
        "naive_bayes": GaussianNB(),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        # --- Regression ---
        "linear_regression": LinearRegression(n_jobs=-1),
        "ridge": Ridge(random_state=random_state),
        "lasso": Lasso(random_state=random_state, max_iter=5000),
        "decision_tree_regressor": DecisionTreeRegressor(random_state=random_state),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=random_state),
        "svr": SVR(),
        # Time-series ML variants (same estimators, feature engineering done upstream)
        "random_forest_ts": RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting_ts": GradientBoostingRegressor(random_state=random_state),
        # --- Clustering ---
        "kmeans": KMeans(n_init="auto", random_state=random_state),
        "dbscan": DBSCAN(),
        "agglomerative": AgglomerativeClustering(),
        "gmm": GaussianMixture(random_state=random_state),
        # --- Anomaly detection ---
        "isolation_forest": IsolationForest(random_state=random_state, n_jobs=-1),
        "lof": LocalOutlierFactor(novelty=True, n_jobs=-1),
        "one_class_svm": OneClassSVM(),
        # --- NLP (TF-IDF vectoriser added in train_model) ---
        "tfidf_logreg": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1,
        ),
        "tfidf_rf": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    # XGBoost models
    if _XGBOOST_AVAILABLE:
        registry["xgboost"] = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        registry["xgboost_regressor"] = xgb.XGBRegressor(
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_id.startswith("xgboost"):
        raise ValueError(
            f"Model '{model_id}' requires XGBoost. Install it with: pip install xgboost"
        )

    # LightGBM models
    if _LIGHTGBM_AVAILABLE:
        registry["lightgbm"] = lgb.LGBMClassifier(random_state=random_state, n_jobs=-1)
        registry["lightgbm_regressor"] = lgb.LGBMRegressor(random_state=random_state, n_jobs=-1)
    elif model_id.startswith("lightgbm"):
        raise ValueError(
            f"Model '{model_id}' requires LightGBM. Install it with: pip install lightgbm"
        )

    if model_id not in registry:
        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Supported IDs: {sorted(registry.keys())}"
        )

    return registry[model_id]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _extract_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str] | None = None,
) -> dict[str, float] | None:
    """Extract feature importances from the last step of a pipeline, if available."""
    estimator = pipeline.named_steps.get("model")
    if estimator is None:
        return None

    importances: np.ndarray | None = None

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef)

    if importances is None:
        return None

    if feature_names and len(feature_names) == len(importances):
        return dict(zip(feature_names, importances.tolist()))

    return {f"feature_{i}": float(v) for i, v in enumerate(importances)}


def train_model(
    model_id: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    preprocessor: Any,
    task_type: str,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a single model and return a results dictionary.

    Parameters
    ----------
    model_id : str
        Snake-case model identifier.
    X_train : pd.DataFrame
        Training feature dataframe (pre-split, pre-preprocessing).
    y_train : pd.Series or np.ndarray
        Training target.
    preprocessor : ColumnTransformer
        Unfitted sklearn ColumnTransformer.
    task_type : str
        Task type string.
    random_state : int, optional
        Random seed (default 42).

    Returns
    -------
    dict
        Keys: ``model_id``, ``pipeline``, ``model``, ``training_time``.
    """
    estimator = get_sklearn_model(model_id, task_type, random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    start_time = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    logger.info("Trained %s in %.2fs", model_id, training_time)

    return {
        "model_id": model_id,
        "pipeline": pipeline,
        "model": pipeline.named_steps["model"],
        "training_time": training_time,
    }


def _run_evaluation(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    profile: dict[str, Any],
    model_ids: list[str],
    cv: int | None,
    random_state: int,
) -> list[dict[str, Any]]:
    """Core evaluation loop shared by quick/standard/advanced modes.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_col : str
        Target column name.
    task_type : str
        Task type string.
    profile : dict
        Dataset profile from the profiler module.
    model_ids : list[str]
        Model IDs to evaluate.
    cv : int or None
        Cross-validation folds.  ``None`` disables cross-validation.
    random_state : int
        Random seed.

    Returns
    -------
    list[dict]
        List of result dictionaries — one per model.
    """
    # Emit warnings for small or imbalanced datasets
    n_samples = len(df)
    if n_samples < 100:
        warnings.warn(
            f"Dataset has only {n_samples} samples. Results may be unreliable.",
            UserWarning,
            stacklevel=3,
        )

    if task_type.lower() == "classification":
        target_counts = df[target_col].value_counts()
        if len(target_counts) >= 2:
            imbalance_ratio = target_counts.iloc[0] / target_counts.iloc[-1]
            if imbalance_ratio > 5:
                warnings.warn(
                    f"Class imbalance detected (ratio {imbalance_ratio:.1f}:1). "
                    "Consider using class_weight='balanced' or resampling.",
                    UserWarning,
                    stacklevel=3,
                )

    # Feature column inference from profile (with fallback to direct dtype inference)
    col_types = profile.get("column_types", {})
    numerical_cols = profile.get("numerical_cols") or col_types.get("numerical")
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numerical_cols = list(numerical_cols)

    categorical_cols = profile.get("categorical_cols") or col_types.get("categorical")
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    else:
        categorical_cols = list(categorical_cols)

    # Remove target from feature lists if accidentally included
    numerical_cols = [c for c in numerical_cols if c != target_col and c in df.columns]
    categorical_cols = [c for c in categorical_cols if c != target_col and c in df.columns]

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(
        df, target_col, task_type, numerical_cols, categorical_cols
    )

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df, target_col, task_type, random_state=random_state
    )

    # Encode target
    y_train_enc, encoder = encode_target(y_train, task_type)
    y_test_enc, _ = encode_target(y_test, task_type)

    results: list[dict[str, Any]] = []

    for model_id in model_ids:
        logger.info("Evaluating model: %s", model_id)

        try:
            import copy
            prep_copy = copy.deepcopy(preprocessor)
            train_result = train_model(
                model_id, X_train, y_train_enc, prep_copy, task_type, random_state
            )
        except Exception as exc:
            logger.error("Failed to train %s: %s", model_id, exc, exc_info=True)
            results.append(
                {
                    "model_id": model_id,
                    "model_name": model_id,
                    "error": str(exc),
                    "metrics": {},
                    "training_time": 0.0,
                    "pipeline": None,
                    "y_pred": None,
                    "y_test": y_test_enc,
                    "y_pred_proba": None,
                    "feature_importance": None,
                    "cv_scores": None,
                }
            )
            continue

        pipeline = train_result["pipeline"]

        # Predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = pipeline.predict(X_test)

        y_pred_proba: np.ndarray | None = None
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            try:
                y_pred_proba = pipeline.predict_proba(X_test)
            except Exception:
                pass

        # Metrics
        metrics = compute_metrics(y_test_enc, y_pred, task_type, y_pred_proba=y_pred_proba)

        # Feature importance
        feature_importance = _extract_feature_importance(pipeline)

        # Cross-validation
        cv_scores: dict | None = None
        if cv is not None and cv > 1:
            from sklearn.model_selection import cross_validate

            primary_metric = get_primary_metric(task_type)
            try:
                import copy
                prep_cv = copy.deepcopy(preprocessor)
                cv_pipeline = Pipeline(
                    steps=[
                        ("preprocessor", prep_cv),
                        ("model", get_sklearn_model(model_id, task_type, random_state)),
                    ]
                )
                X_full = df.drop(columns=[target_col])
                y_full_enc, _ = encode_target(df[target_col], task_type)
                scoring = _resolve_cv_scoring(task_type, primary_metric)
                cv_result = cross_validate(
                    cv_pipeline,
                    X_full,
                    y_full_enc,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=False,
                )
                cv_scores = {
                    "mean": float(np.mean(cv_result[f"test_{scoring}"])),
                    "std": float(np.std(cv_result[f"test_{scoring}"])),
                    "scores": cv_result[f"test_{scoring}"].tolist(),
                    "metric": scoring,
                }
            except Exception as exc:
                logger.warning("Cross-validation failed for %s: %s", model_id, exc)
                cv_scores = {"error": str(exc)}

        # Resolve display name
        model_name = _resolve_model_name(model_id)

        results.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "metrics": metrics,
                "training_time": train_result["training_time"],
                "pipeline": pipeline,
                "y_pred": y_pred,
                "y_test": y_test_enc,
                "y_pred_proba": y_pred_proba,
                "feature_importance": feature_importance,
                "cv_scores": cv_scores,
            }
        )

    return results


def _resolve_cv_scoring(task_type: str, primary_metric: str) -> str:
    """Map primary metric name to sklearn scoring string."""
    mapping = {
        "f1": "f1_weighted",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    return mapping.get(primary_metric, "accuracy")


def _resolve_model_name(model_id: str) -> str:
    """Return a human-readable model name for the given model ID."""
    name_map = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "svm": "Support Vector Machine",
        "knn": "K-Nearest Neighbours",
        "naive_bayes": "Naive Bayes",
        "gradient_boosting": "Gradient Boosting",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "linear_regression": "Linear Regression",
        "ridge": "Ridge Regression",
        "lasso": "Lasso Regression",
        "decision_tree_regressor": "Decision Tree Regressor",
        "random_forest_regressor": "Random Forest Regressor",
        "gradient_boosting_regressor": "Gradient Boosting Regressor",
        "xgboost_regressor": "XGBoost Regressor",
        "lightgbm_regressor": "LightGBM Regressor",
        "svr": "Support Vector Regressor",
        "random_forest_ts": "Random Forest (Time-Series)",
        "gradient_boosting_ts": "Gradient Boosting (Time-Series)",
        "kmeans": "K-Means",
        "dbscan": "DBSCAN",
        "agglomerative": "Agglomerative Clustering",
        "gmm": "Gaussian Mixture Model",
        "isolation_forest": "Isolation Forest",
        "lof": "Local Outlier Factor",
        "one_class_svm": "One-Class SVM",
        "tfidf_logreg": "TF-IDF + Logistic Regression",
        "tfidf_rf": "TF-IDF + Random Forest",
    }
    return name_map.get(model_id, model_id.replace("_", " ").title())


def _get_quick_model_ids(task_type: str, profile: dict[str, Any]) -> list[str]:
    """Return 3 model IDs for the quick evaluation mode."""
    quick_map: dict[str, list[str]] = {
        "classification": ["logistic_regression", "random_forest", "gradient_boosting"],
        "regression": ["linear_regression", "random_forest_regressor", "gradient_boosting_regressor"],
        "clustering": ["kmeans", "dbscan", "agglomerative"],
        "time_series": ["random_forest_ts", "gradient_boosting_ts", "arima"],
        "nlp": ["tfidf_logreg", "tfidf_rf"],
        "anomaly_detection": ["isolation_forest", "lof", "one_class_svm"],
    }
    return quick_map.get(task_type.lower(), ["logistic_regression"])[:3]


# ---------------------------------------------------------------------------
# Public evaluation API
# ---------------------------------------------------------------------------

def evaluate_quick(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    profile: dict[str, Any],
    model_ids: list[str] | None = None,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Train and evaluate 3 models — suitable for a rapid first assessment.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target column.
    target_col : str
        Name of the target column.
    task_type : str
        Task type string.
    profile : dict
        Dataset profile produced by the profiler module.
    model_ids : list[str] | None, optional
        Override the default model IDs.  If ``None``, 3 sensible defaults are used.
    random_state : int, optional
        Random seed (default 42).

    Returns
    -------
    list[dict]
        List of result dicts (one per model).
    """
    ids = model_ids if model_ids else _get_quick_model_ids(task_type, profile)
    logger.info("Quick evaluation: %s", ids)
    return _run_evaluation(df, target_col, task_type, profile, ids, cv=None, random_state=random_state)


def evaluate_standard(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    profile: dict[str, Any],
    model_ids: list[str] | None = None,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Train and evaluate the full candidate list without cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target column.
    target_col : str
        Name of the target column.
    task_type : str
        Task type string.
    profile : dict
        Dataset profile produced by the profiler module.
    model_ids : list[str] | None, optional
        Override the default model IDs.  If ``None``, all candidates for the
        task type are used.
    random_state : int, optional
        Random seed (default 42).

    Returns
    -------
    list[dict]
        List of result dicts (one per model).
    """
    if model_ids:
        ids = model_ids
    else:
        candidates = get_model_candidates(task_type, profile)
        ids = [c["id"] for c in candidates]

    logger.info("Standard evaluation: %s", ids)
    return _run_evaluation(df, target_col, task_type, profile, ids, cv=None, random_state=random_state)


def evaluate_advanced(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    profile: dict[str, Any],
    model_ids: list[str] | None = None,
    cv: int = 5,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Train and evaluate models with k-fold cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target column.
    target_col : str
        Name of the target column.
    task_type : str
        Task type string.
    profile : dict
        Dataset profile produced by the profiler module.
    model_ids : list[str] | None, optional
        Override the default model IDs.  If ``None``, all candidates for the
        task type are used.
    cv : int, optional
        Number of cross-validation folds (default 5).
    random_state : int, optional
        Random seed (default 42).

    Returns
    -------
    list[dict]
        List of result dicts (one per model), each including a ``cv_scores`` key.
    """
    if model_ids:
        ids = model_ids
    else:
        candidates = get_model_candidates(task_type, profile)
        ids = [c["id"] for c in candidates]

    logger.info("Advanced evaluation (cv=%d): %s", cv, ids)
    return _run_evaluation(
        df, target_col, task_type, profile, ids, cv=cv, random_state=random_state
    )
