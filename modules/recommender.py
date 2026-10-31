"""
recommender.py
==============
Model recommendation engine for the AutoML Model Selector.

Provides intelligent model candidate selection based on dataset characteristics
(task type, number of samples, number of features, etc.) and generates detailed
explanations for each recommendation.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Internal catalogue helpers
# ---------------------------------------------------------------------------

def _classification_candidates(profile: dict[str, Any], sub_type: str | None) -> list[dict]:
    """Return full classification model catalogue."""
    n_samples: int = profile.get("n_samples", 1000)
    n_features: int = profile.get("n_features", 10)
    has_missing: bool = profile.get("has_missing", False)
    has_categorical: bool = bool(profile.get("categorical_cols"))
    n_classes: int = profile.get("n_classes", 2)
    is_binary: bool = n_classes == 2

    candidates = [
        {
            "name": "Logistic Regression",
            "id": "logistic_regression",
            "category": "baseline",
            "description": "Linear classifier that models the probability of class membership using the logistic function.",
            "why_selected": (
                f"Ideal starting baseline for your dataset with {n_samples} samples and {n_features} features. "
                "Provides interpretable coefficients and fast training."
            ),
            "advantages": [
                "Highly interpretable — coefficients directly indicate feature importance",
                "Very fast to train and predict",
                "Works well with linearly separable data",
                "Probabilistic output (calibrated probabilities)",
                "Low memory footprint",
            ],
            "limitations": [
                "Assumes linear decision boundary",
                "Sensitive to feature scale and outliers",
                "Struggles with complex non-linear relationships",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Data is linearly separable",
                "Interpretability is required",
                "Quick baseline is needed",
                "Low-latency inference is required",
            ],
            "preprocessing_required": [
                "Feature scaling (StandardScaler recommended)",
                "One-hot encoding for categorical variables",
                "Imputation for missing values",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "is_recommended": False,
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Decision Tree",
            "id": "decision_tree",
            "category": "traditional_ml",
            "description": "Recursive partitioning algorithm that builds a tree of if-else rules.",
            "why_selected": (
                f"Good for understanding feature interactions in your {n_features}-feature dataset. "
                "Provides built-in feature importance."
            ),
            "advantages": [
                "Highly interpretable — can be visualised as a tree",
                "No feature scaling required",
                "Handles mixed data types natively",
                "Captures non-linear relationships",
            ],
            "limitations": [
                "Prone to overfitting without pruning",
                "Unstable — small data changes can produce very different trees",
                "Generally lower accuracy than ensembles",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Interpretability is critical",
                "Data contains complex non-linear patterns",
                "Mixed numerical and categorical features",
            ],
            "preprocessing_required": [
                "Ordinal encoding for categorical variables",
                "Optional imputation for missing values",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "is_recommended": False,
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Random Forest",
            "id": "random_forest",
            "category": "ensemble",
            "description": "Ensemble of decision trees trained on random subsets of data and features.",
            "why_selected": (
                f"Strong all-rounder for your dataset size of {n_samples} samples. "
                "Robust to overfitting and provides reliable feature importance."
            ),
            "advantages": [
                "Excellent out-of-the-box performance",
                "Robust to outliers and noise",
                "Built-in feature importance",
                "Handles high-dimensional data well",
                "Parallelisable training",
            ],
            "limitations": [
                "Less interpretable than single decision tree",
                "Higher memory usage than linear models",
                "Slower prediction than simple models",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "General-purpose classification tasks",
                "Dataset has mixed feature types",
                "Feature importance is needed",
                "Robustness to hyperparameter choice is desired",
            ],
            "preprocessing_required": [
                "Ordinal/label encoding for categorical variables",
                "Optional imputation (can handle NaN via workaround)",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "is_recommended": False,
            "min_samples": 100,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Support Vector Machine",
            "id": "svm",
            "category": "traditional_ml",
            "description": "Finds the maximum-margin hyperplane separating classes; kernels enable non-linear boundaries.",
            "why_selected": (
                f"Effective for datasets like yours with {n_features} features. "
                "Kernel trick enables complex decision boundaries."
            ),
            "advantages": [
                "Effective in high-dimensional spaces",
                "Memory efficient (uses only support vectors)",
                "Versatile via kernel choice (RBF, linear, polynomial)",
            ],
            "limitations": [
                "Slow training on large datasets (O(n²)–O(n³))",
                "Requires careful feature scaling",
                "Sensitive to hyperparameter C and gamma",
                "No native probability estimates without Platt scaling",
            ],
            "computational_cost": "high" if n_samples > 5000 else "medium",
            "suitable_when": [
                "Dataset is small to medium sized",
                "Features are well-scaled",
                "High-dimensional sparse data (text classification)",
            ],
            "preprocessing_required": [
                "Feature scaling (StandardScaler mandatory)",
                "One-hot or ordinal encoding for categoricals",
                "Imputation for missing values",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "is_recommended": False,
            "min_samples": 100,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "K-Nearest Neighbours",
            "id": "knn",
            "category": "traditional_ml",
            "description": "Classifies based on the majority class of the k closest training samples.",
            "why_selected": (
                "Non-parametric approach useful when decision boundaries are highly irregular."
            ),
            "advantages": [
                "No training phase — lazy learner",
                "Naturally handles multi-class problems",
                "Can capture very complex decision boundaries",
            ],
            "limitations": [
                "Slow prediction on large datasets (O(n) per query)",
                "Sensitive to irrelevant features and scale",
                "High memory usage (stores all training data)",
                "Performance degrades in high dimensions (curse of dimensionality)",
            ],
            "computational_cost": "low" if n_samples < 1000 else "high",
            "suitable_when": [
                "Dataset is small",
                "Decision boundary is complex and irregular",
                "Low-latency training is required",
            ],
            "preprocessing_required": [
                "Feature scaling (mandatory)",
                "Dimensionality reduction recommended for high-dimensional data",
                "Imputation for missing values",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "is_recommended": False,
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Naive Bayes",
            "id": "naive_bayes",
            "category": "baseline",
            "description": "Probabilistic classifier based on Bayes' theorem with feature independence assumption.",
            "why_selected": (
                f"Fast probabilistic baseline, particularly strong for text-like features "
                f"in your {n_features}-feature dataset."
            ),
            "advantages": [
                "Extremely fast training and prediction",
                "Works well with small datasets",
                "Naturally handles multi-class classification",
                "Robust to irrelevant features",
            ],
            "limitations": [
                "Strong independence assumption rarely holds in practice",
                "Poor probability estimates when assumption is violated",
                "Struggles with feature interactions",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Text or high-dimensional sparse data",
                "Very limited training data",
                "Real-time prediction required",
                "Features are approximately independent",
            ],
            "preprocessing_required": [
                "Feature scaling not required for GaussianNB",
                "Encoding for categorical variables",
                "Imputation for missing values",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "is_recommended": False,
            "min_samples": 20,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Gradient Boosting",
            "id": "gradient_boosting",
            "category": "ensemble",
            "description": "Sequentially builds an ensemble of weak learners (trees), each correcting predecessor errors.",
            "why_selected": (
                f"High-accuracy ensemble method well-suited to your {n_samples}-sample dataset. "
                "Often top performer on tabular data."
            ),
            "advantages": [
                "State-of-the-art performance on tabular data",
                "Built-in feature importance",
                "Handles mixed feature types well",
                "Robust to outliers with appropriate loss",
            ],
            "limitations": [
                "Slower training than Random Forest",
                "More hyperparameters to tune",
                "Prone to overfitting without regularisation",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "High predictive performance is the priority",
                "Dataset is tabular with mixed types",
                "Sufficient compute time is available for tuning",
            ],
            "preprocessing_required": [
                "Ordinal encoding for categorical variables",
                "Optional imputation",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "is_recommended": False,
            "min_samples": 200,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "XGBoost",
            "id": "xgboost",
            "category": "ensemble",
            "description": "Optimised gradient boosting library with regularisation, sparse-aware algorithms and GPU support.",
            "why_selected": (
                f"Industry-proven winner on tabular competitions. Excellent fit for your "
                f"{n_samples}-sample, {n_features}-feature dataset."
            ),
            "advantages": [
                "Top performance on structured/tabular data",
                "Native handling of missing values",
                "Built-in L1/L2 regularisation",
                "Supports GPU acceleration",
                "Fast training via parallel tree construction",
            ],
            "limitations": [
                "Many hyperparameters (requires tuning for best results)",
                "Less interpretable than single models",
                "Requires separate installation (pip install xgboost)",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "Maximum predictive performance is needed",
                "Dataset is tabular",
                "Missing values are present",
                "GPU resources are available",
            ],
            "preprocessing_required": [
                "Label/ordinal encoding for categorical variables",
                "Missing values handled natively",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "is_recommended": False,
            "min_samples": 200,
            "handles_missing": True,
            "handles_categorical": False,
        },
    ]

    # Filter by minimum sample count
    candidates = [c for c in candidates if n_samples >= c["min_samples"]]

    # Mark recommendation
    candidates = _mark_classification_recommendation(candidates, n_samples, n_features, is_binary, has_missing)

    return candidates


def _mark_classification_recommendation(
    candidates: list[dict],
    n_samples: int,
    n_features: int,
    is_binary: bool,
    has_missing: bool,
) -> list[dict]:
    """Apply heuristic recommendation logic for classification."""
    preferred_id: str

    if n_samples < 200:
        preferred_id = "naive_bayes"
    elif n_samples < 1000:
        preferred_id = "random_forest"
    elif n_samples < 10_000:
        preferred_id = "gradient_boosting"
    else:
        preferred_id = "xgboost"

    # XGBoost preferred when missing values present (handles them natively)
    if has_missing and any(c["id"] == "xgboost" for c in candidates):
        preferred_id = "xgboost"

    for c in candidates:
        c["is_recommended"] = c["id"] == preferred_id

    return candidates


def _regression_candidates(profile: dict[str, Any]) -> list[dict]:
    """Return full regression model catalogue."""
    n_samples: int = profile.get("n_samples", 1000)
    n_features: int = profile.get("n_features", 10)
    has_missing: bool = profile.get("has_missing", False)

    candidates = [
        {
            "name": "Linear Regression",
            "id": "linear_regression",
            "category": "baseline",
            "description": "Ordinary least-squares linear model — the standard regression baseline.",
            "why_selected": (
                f"Essential baseline for your {n_samples}-sample regression task. "
                "Extremely fast and interpretable."
            ),
            "advantages": [
                "Highly interpretable coefficients",
                "Very fast training and inference",
                "No hyperparameters",
                "Well-understood statistical properties",
            ],
            "limitations": [
                "Assumes linear relationship between features and target",
                "Sensitive to outliers",
                "Multicollinearity can destabilise coefficients",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Relationship between features and target is approximately linear",
                "Interpretability is critical",
                "Baseline performance is needed quickly",
            ],
            "preprocessing_required": [
                "Feature scaling recommended",
                "One-hot encoding for categoricals",
                "Imputation for missing values",
            ],
            "metrics": ["mae", "mse", "rmse", "r2"],
            "is_recommended": False,
            "min_samples": 20,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Ridge Regression",
            "id": "ridge",
            "category": "traditional_ml",
            "description": "Linear regression with L2 regularisation to reduce overfitting and handle multicollinearity.",
            "why_selected": (
                f"Regularised linear model well-suited to your {n_features}-feature dataset. "
                "More robust than plain OLS when features are correlated."
            ),
            "advantages": [
                "Handles multicollinearity gracefully",
                "L2 regularisation prevents overfitting",
                "Closed-form solution — very fast training",
                "Interpretable coefficients",
            ],
            "limitations": [
                "Still assumes linear relationship",
                "Does not perform feature selection (keeps all features)",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Features are correlated",
                "Dataset is high-dimensional",
                "Slight regularisation is preferred over OLS",
            ],
            "preprocessing_required": [
                "Feature scaling (mandatory for regularisation to work correctly)",
                "One-hot encoding for categoricals",
                "Imputation for missing values",
            ],
            "metrics": ["mae", "mse", "rmse", "r2"],
            "is_recommended": False,
            "min_samples": 20,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Lasso Regression",
            "id": "lasso",
            "category": "traditional_ml",
            "description": "Linear regression with L1 regularisation that performs automatic feature selection.",
            "why_selected": (
                f"Useful when many of the {n_features} features may be irrelevant. "
                "L1 penalty drives irrelevant coefficients to exactly zero."
            ),
            "advantages": [
                "Automatic feature selection via L1 sparsity",
                "Interpretable sparse model",
                "Prevents overfitting",
            ],
            "limitations": [
                "Can be unstable when features are highly correlated",
                "May under-select features with small but real effects",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Feature selection is desired",
                "Sparse model is preferred",
                "Many features may be irrelevant",
            ],
            "preprocessing_required": [
                "Feature scaling (mandatory)",
                "One-hot encoding for categoricals",
                "Imputation for missing values",
            ],
            "metrics": ["mae", "mse", "rmse", "r2"],
            "is_recommended": False,
            "min_samples": 20,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Decision Tree Regressor",
            "id": "decision_tree_regressor",
            "category": "traditional_ml",
            "description": "Regression tree that recursively partitions feature space to minimise prediction error.",
            "why_selected": "Captures non-linear feature interactions without feature scaling.",
            "advantages": [
                "Handles non-linear relationships",
                "No feature scaling required",
                "Easily interpretable",
            ],
            "limitations": [
                "High variance — prone to overfitting",
                "Piecewise constant predictions",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Non-linear target relationships",
                "Mixed feature types",
                "Interpretability needed",
            ],
            "preprocessing_required": [
                "Ordinal encoding for categoricals",
                "Optional imputation",
            ],
            "metrics": ["mae", "mse", "rmse", "r2"],
            "is_recommended": False,
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Random Forest Regressor",
            "id": "random_forest_regressor",
            "category": "ensemble",
            "description": "Ensemble of regression trees — averages predictions to reduce variance.",
            "why_selected": (
                f"Robust and accurate all-rounder for your {n_samples}-sample regression task."
            ),
            "advantages": [
                "Low variance compared to single tree",
                "Feature importance estimation",
                "Handles high-dimensional data",
                "Robust to outliers",
            ],
            "limitations": [
                "Less interpretable than single tree or linear models",
                "Slower prediction than linear models",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "General regression tasks",
                "Feature importance needed",
                "Non-linear target relationships",
            ],
            "preprocessing_required": [
                "Ordinal encoding for categoricals",
                "Optional imputation",
            ],
            "metrics": ["mae", "mse", "rmse", "r2"],
            "is_recommended": False,
            "min_samples": 100,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Gradient Boosting Regressor",
            "id": "gradient_boosting_regressor",
            "category": "ensemble",
            "description": "Sequential boosting ensemble that minimises a differentiable loss function.",
            "why_selected": (
                f"High-accuracy method for your {n_samples}-sample dataset. "
                "Often outperforms Random Forest on tabular regression."
            ),
            "advantages": [
                "State-of-the-art accuracy on tabular data",
                "Flexible loss functions",
                "Built-in feature importance",
            ],
            "limitations": [
                "Slower to train than Random Forest",
                "More hyperparameters to tune",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "High accuracy is the priority",
                "Tabular data with mixed types",
            ],
            "preprocessing_required": [
                "Ordinal encoding for categoricals",
                "Optional imputation",
            ],
            "metrics": ["mae", "mse", "rmse", "r2"],
            "is_recommended": False,
            "min_samples": 200,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Support Vector Regressor",
            "id": "svr",
            "category": "traditional_ml",
            "description": "Regression variant of SVM using the epsilon-insensitive loss and kernel trick.",
            "why_selected": "Effective for small datasets with complex non-linear relationships.",
            "advantages": [
                "Effective in high-dimensional spaces",
                "Kernel trick enables non-linear regression",
                "Robust to outliers within the epsilon tube",
            ],
            "limitations": [
                "Slow training for large datasets",
                "Requires careful feature scaling",
                "Sensitive to hyperparameter choice",
            ],
            "computational_cost": "high" if n_samples > 5000 else "medium",
            "suitable_when": [
                "Small to medium datasets",
                "Non-linear target relationships",
                "Features are well-scaled",
            ],
            "preprocessing_required": [
                "Feature scaling (mandatory)",
                "One-hot/ordinal encoding for categoricals",
                "Imputation for missing values",
            ],
            "metrics": ["mae", "mse", "rmse", "r2"],
            "is_recommended": False,
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
        },
    ]

    candidates = [c for c in candidates if n_samples >= c["min_samples"]]
    candidates = _mark_regression_recommendation(candidates, n_samples, n_features, has_missing)
    return candidates


def _mark_regression_recommendation(
    candidates: list[dict],
    n_samples: int,
    n_features: int,
    has_missing: bool,
) -> list[dict]:
    """Apply heuristic recommendation logic for regression."""
    if n_samples < 200:
        preferred_id = "ridge"
    elif n_samples < 1000:
        preferred_id = "random_forest_regressor"
    else:
        preferred_id = "gradient_boosting_regressor"

    for c in candidates:
        c["is_recommended"] = c["id"] == preferred_id

    return candidates


def _clustering_candidates(profile: dict[str, Any]) -> list[dict]:
    """Return clustering model catalogue."""
    n_samples: int = profile.get("n_samples", 1000)
    n_features: int = profile.get("n_features", 10)

    candidates = [
        {
            "name": "K-Means",
            "id": "kmeans",
            "category": "traditional_ml",
            "description": "Partitions data into k clusters by iteratively minimising within-cluster variance.",
            "why_selected": (
                f"Fast and scalable baseline for clustering your {n_samples}-sample dataset. "
                "Works well when clusters are roughly spherical and equal-sized."
            ),
            "advantages": [
                "Very fast and scalable (O(n·k·i))",
                "Simple to understand and implement",
                "Works well for compact spherical clusters",
                "Supports mini-batch variant for very large data",
            ],
            "limitations": [
                "Must specify number of clusters k upfront",
                "Sensitive to outliers and initialisation",
                "Assumes spherical, equal-variance clusters",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Number of clusters is known approximately",
                "Clusters are roughly spherical",
                "Dataset is large",
            ],
            "preprocessing_required": [
                "Feature scaling (mandatory)",
                "Dimensionality reduction recommended for high-dimensional data",
                "Imputation for missing values",
            ],
            "metrics": ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"],
            "is_recommended": False,
            "min_samples": 10,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "DBSCAN",
            "id": "dbscan",
            "category": "traditional_ml",
            "description": "Density-based clustering — discovers arbitrarily shaped clusters and marks outliers as noise.",
            "why_selected": (
                "Excellent choice when cluster shapes are unknown and outlier detection is desired."
            ),
            "advantages": [
                "Does not require specifying number of clusters",
                "Identifies outliers as noise points",
                "Discovers arbitrarily shaped clusters",
                "Robust to outliers",
            ],
            "limitations": [
                "Sensitive to epsilon and min_samples hyperparameters",
                "Struggles with varying-density clusters",
                "Slow on very large datasets (O(n log n) with index)",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "Number of clusters is unknown",
                "Clusters have irregular shapes",
                "Outlier detection is needed alongside clustering",
            ],
            "preprocessing_required": [
                "Feature scaling (mandatory)",
                "Dimensionality reduction for high-dimensional data",
                "Imputation for missing values",
            ],
            "metrics": ["silhouette_score", "davies_bouldin_score"],
            "is_recommended": False,
            "min_samples": 10,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Agglomerative Clustering",
            "id": "agglomerative",
            "category": "traditional_ml",
            "description": "Hierarchical bottom-up clustering that merges the closest pair of clusters at each step.",
            "why_selected": (
                "Useful when hierarchical structure in the data is expected or interpretability of cluster merges is needed."
            ),
            "advantages": [
                "No need to specify k in advance (with dendrogram)",
                "Produces interpretable hierarchy (dendrogram)",
                "Flexible linkage criteria (ward, complete, average)",
                "Deterministic — no random initialisation",
            ],
            "limitations": [
                "O(n² log n) memory and time complexity — slow for large datasets",
                "Cannot reassign points once merged",
            ],
            "computational_cost": "high" if n_samples > 5000 else "medium",
            "suitable_when": [
                "Dataset is small to medium",
                "Hierarchical cluster structure is interesting",
                "Deterministic results required",
            ],
            "preprocessing_required": [
                "Feature scaling recommended",
                "Imputation for missing values",
            ],
            "metrics": ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"],
            "is_recommended": False,
            "min_samples": 10,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Gaussian Mixture Model",
            "id": "gmm",
            "category": "traditional_ml",
            "description": "Probabilistic model assuming data is generated from a mixture of Gaussian distributions.",
            "why_selected": (
                "Soft cluster assignments and probabilistic framework make it suitable "
                "when cluster boundaries are fuzzy."
            ),
            "advantages": [
                "Soft cluster assignments (probabilistic membership)",
                "Handles elliptical cluster shapes",
                "Supports different covariance structures",
                "Can estimate cluster density",
            ],
            "limitations": [
                "Requires specifying number of components",
                "Sensitive to initialisation — may converge to local optimum",
                "Assumes Gaussian-distributed clusters",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "Soft/overlapping cluster boundaries expected",
                "Cluster densities need to be modelled",
                "Data is approximately Gaussian",
            ],
            "preprocessing_required": [
                "Feature scaling (recommended)",
                "Imputation for missing values",
            ],
            "metrics": ["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"],
            "is_recommended": False,
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
        },
    ]

    candidates = [c for c in candidates if n_samples >= c["min_samples"]]

    # Recommendation heuristic for clustering
    preferred_id = "kmeans" if n_samples > 500 else "agglomerative"
    for c in candidates:
        c["is_recommended"] = c["id"] == preferred_id

    return candidates


def _time_series_candidates(profile: dict[str, Any]) -> list[dict]:
    """Return time-series model catalogue."""
    n_samples: int = profile.get("n_samples", 1000)

    candidates = [
        {
            "name": "ARIMA",
            "id": "arima",
            "category": "baseline",
            "description": "AutoRegressive Integrated Moving Average — classical statistical time-series model.",
            "why_selected": "Standard baseline for univariate time-series forecasting.",
            "advantages": [
                "Well-understood statistical model",
                "Interpretable parameters (p, d, q)",
                "Works well on stationary or differenced series",
            ],
            "limitations": [
                "Assumes linear relationships",
                "Univariate only (standard ARIMA)",
                "Requires stationarity",
                "Manual order selection can be tedious",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Univariate forecasting",
                "Series shows autocorrelation",
                "Statistical interpretability required",
            ],
            "preprocessing_required": [
                "Stationarity check and differencing",
                "Outlier treatment",
            ],
            "metrics": ["mae", "mse", "rmse", "mape"],
            "is_recommended": False,
            "min_samples": 30,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Random Forest (Time-Series Features)",
            "id": "random_forest_ts",
            "category": "ensemble",
            "description": "Random Forest trained on engineered time-series lag features.",
            "why_selected": (
                f"Flexible ML approach for your {n_samples}-step time series. "
                "Feature engineering captures temporal patterns."
            ),
            "advantages": [
                "Can incorporate exogenous variables easily",
                "Handles non-linear temporal patterns",
                "No stationarity requirement",
            ],
            "limitations": [
                "Requires manual lag feature engineering",
                "Does not extrapolate beyond training range",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "Multivariate time series",
                "External regressors available",
                "Non-linear seasonal patterns",
            ],
            "preprocessing_required": [
                "Lag feature engineering",
                "Rolling statistics features",
                "Imputation for missing values",
            ],
            "metrics": ["mae", "mse", "rmse", "mape"],
            "is_recommended": True,
            "min_samples": 100,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Gradient Boosting (Time-Series Features)",
            "id": "gradient_boosting_ts",
            "category": "ensemble",
            "description": "Gradient Boosting Regressor trained on engineered time-series lag and calendar features.",
            "why_selected": "High-accuracy tabular ML approach to time-series forecasting.",
            "advantages": [
                "State-of-the-art accuracy with good feature engineering",
                "Handles complex non-linear seasonality",
                "Can use many exogenous variables",
            ],
            "limitations": [
                "Requires careful feature engineering",
                "Cannot extrapolate beyond training data range",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "High forecasting accuracy required",
                "Rich exogenous features available",
            ],
            "preprocessing_required": [
                "Lag and rolling feature engineering",
                "Calendar feature extraction",
            ],
            "metrics": ["mae", "mse", "rmse", "mape"],
            "is_recommended": False,
            "min_samples": 200,
            "handles_missing": False,
            "handles_categorical": False,
        },
    ]

    candidates = [c for c in candidates if n_samples >= c["min_samples"]]
    return candidates


def _nlp_candidates(profile: dict[str, Any]) -> list[dict]:
    """Return NLP model catalogue."""
    n_samples: int = profile.get("n_samples", 1000)

    candidates = [
        {
            "name": "TF-IDF + Logistic Regression",
            "id": "tfidf_logreg",
            "category": "baseline",
            "description": "Classic NLP pipeline: TF-IDF vectorisation followed by logistic regression classifier.",
            "why_selected": "Fast and effective NLP baseline with good performance on most text classification tasks.",
            "advantages": [
                "Very fast training and inference",
                "Interpretable features (top TF-IDF tokens per class)",
                "Strong baseline for binary and multi-class text classification",
            ],
            "limitations": [
                "Ignores word order and semantic meaning",
                "Sparse representation may miss contextual relationships",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Text classification tasks",
                "Quick baseline required",
                "Limited compute resources",
            ],
            "preprocessing_required": [
                "Tokenisation and lowercasing",
                "Stop word removal (optional)",
                "TF-IDF vectorisation",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "is_recommended": True,
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "TF-IDF + Random Forest",
            "id": "tfidf_rf",
            "category": "ensemble",
            "description": "TF-IDF vectorisation combined with Random Forest for robust text classification.",
            "why_selected": "More robust than logistic regression for complex text tasks with non-linear patterns.",
            "advantages": [
                "Handles non-linear text patterns",
                "Feature importance from TF-IDF weights + tree importance",
            ],
            "limitations": [
                "Slower than Logistic Regression",
                "Higher memory usage for large vocabularies",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "Complex text classification with non-linear patterns",
                "Baseline logistic regression underperforms",
            ],
            "preprocessing_required": [
                "Tokenisation",
                "TF-IDF vectorisation",
            ],
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "is_recommended": False,
            "min_samples": 100,
            "handles_missing": False,
            "handles_categorical": False,
        },
    ]

    candidates = [c for c in candidates if n_samples >= c["min_samples"]]
    return candidates


def _anomaly_detection_candidates(profile: dict[str, Any]) -> list[dict]:
    """Return anomaly detection model catalogue."""
    n_samples: int = profile.get("n_samples", 1000)
    n_features: int = profile.get("n_features", 10)

    candidates = [
        {
            "name": "Isolation Forest",
            "id": "isolation_forest",
            "category": "ensemble",
            "description": "Tree-based anomaly detection that isolates anomalies by random feature splits.",
            "why_selected": (
                f"Efficient and effective for your {n_samples}-sample, {n_features}-feature dataset. "
                "No assumption about data distribution."
            ),
            "advantages": [
                "Linear time complexity — scales well",
                "No distributional assumptions",
                "Works well in high dimensions",
                "Handles mixed feature types after encoding",
            ],
            "limitations": [
                "Score interpretation requires threshold selection",
                "May struggle with very high-dimensional data",
            ],
            "computational_cost": "low",
            "suitable_when": [
                "Unsupervised anomaly detection",
                "High-dimensional data",
                "No prior knowledge of anomaly distribution",
            ],
            "preprocessing_required": [
                "Feature scaling recommended",
                "Imputation for missing values",
            ],
            "metrics": ["anomaly_ratio", "silhouette_score"],
            "is_recommended": True,
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "Local Outlier Factor",
            "id": "lof",
            "category": "traditional_ml",
            "description": "Density-based anomaly detection comparing local density of a point to its neighbours.",
            "why_selected": "Effective when anomalies are defined by local density differences rather than global statistics.",
            "advantages": [
                "Detects local anomalies that global methods miss",
                "No distributional assumptions",
                "Works well for clustered data with local density variations",
            ],
            "limitations": [
                "Slow on large datasets (O(n²) naive)",
                "Sensitive to n_neighbors hyperparameter",
                "Not directly usable for prediction on new data (novelty mode needed)",
            ],
            "computational_cost": "medium",
            "suitable_when": [
                "Local anomalies in clustered data",
                "Dataset is small to medium",
            ],
            "preprocessing_required": [
                "Feature scaling (mandatory)",
                "Imputation for missing values",
            ],
            "metrics": ["anomaly_ratio"],
            "is_recommended": False,
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
        },
        {
            "name": "One-Class SVM",
            "id": "one_class_svm",
            "category": "traditional_ml",
            "description": "SVM trained only on normal data; flags deviations as anomalies.",
            "why_selected": "Useful when a clear definition of 'normal' class boundary is needed.",
            "advantages": [
                "Effective in high-dimensional spaces",
                "Kernel trick enables non-linear decision boundary",
            ],
            "limitations": [
                "Slow training for large datasets",
                "Sensitive to kernel choice and hyperparameters",
                "Requires scaled features",
            ],
            "computational_cost": "high" if n_samples > 5000 else "medium",
            "suitable_when": [
                "Small to medium datasets",
                "Normal class is well-defined",
            ],
            "preprocessing_required": [
                "Feature scaling (mandatory)",
                "Imputation for missing values",
            ],
            "metrics": ["anomaly_ratio"],
            "is_recommended": False,
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
        },
    ]

    candidates = [c for c in candidates if n_samples >= c["min_samples"]]
    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_candidates(
    task_type: str,
    profile: dict[str, Any],
    sub_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return a list of model candidate dicts appropriate for the given task type and dataset profile.

    Parameters
    ----------
    task_type : str
        One of 'classification', 'regression', 'clustering', 'time_series', 'nlp',
        'anomaly_detection'.
    profile : dict
        Dataset profile as produced by the data profiler module.  Relevant keys:
        - n_samples (int)
        - n_features (int)
        - has_missing (bool)
        - categorical_cols (list)
        - n_classes (int, classification only)
    sub_type : str | None
        Optional task sub-type for future expansion (e.g. 'multiclass', 'multilabel').

    Returns
    -------
    list[dict]
        List of model candidate dicts.  Exactly one dict will have ``is_recommended=True``.

    Raises
    ------
    ValueError
        If ``task_type`` is not supported.
    """
    task_type = task_type.strip().lower()
    dispatch = {
        "classification": lambda: _classification_candidates(profile, sub_type),
        "regression": lambda: _regression_candidates(profile),
        "clustering": lambda: _clustering_candidates(profile),
        "time_series": lambda: _time_series_candidates(profile),
        "nlp": lambda: _nlp_candidates(profile),
        "anomaly_detection": lambda: _anomaly_detection_candidates(profile),
    }

    if task_type not in dispatch:
        raise ValueError(
            f"Unsupported task_type '{task_type}'. "
            f"Choose one of: {', '.join(dispatch.keys())}"
        )

    return dispatch[task_type]()


def explain_recommendation(candidate: dict[str, Any], profile: dict[str, Any]) -> str:
    """Generate a detailed, human-readable explanation for a model candidate recommendation.

    Parameters
    ----------
    candidate : dict
        A model candidate dict as returned by :func:`get_model_candidates`.
    profile : dict
        Dataset profile dict (same as passed to :func:`get_model_candidates`).

    Returns
    -------
    str
        Multi-line explanation string suitable for display in a terminal or UI.
    """
    n_samples = profile.get("n_samples", "N/A")
    n_features = profile.get("n_features", "N/A")
    has_missing = profile.get("has_missing", False)
    missing_pct = profile.get("missing_percentage", 0.0)

    badge = "⭐ TOP PICK" if candidate.get("is_recommended") else "Candidate"
    lines = [
        f"{'=' * 60}",
        f"{badge}: {candidate['name']} [{candidate['id']}]",
        f"Category: {candidate['category'].replace('_', ' ').title()}",
        f"{'=' * 60}",
        "",
        f"📋 Description",
        f"  {candidate['description']}",
        "",
        f"🎯 Why selected for this dataset",
        f"  {candidate['why_selected']}",
        "",
        f"📊 Dataset context",
        f"  • Samples : {n_samples:,}" if isinstance(n_samples, int) else f"  • Samples : {n_samples}",
        f"  • Features: {n_features}",
        f"  • Missing data: {'Yes' if has_missing else 'No'}" +
        (f" ({missing_pct:.1f}%)" if has_missing and missing_pct else ""),
        "",
        f"✅ Advantages",
    ]
    for adv in candidate.get("advantages", []):
        lines.append(f"  • {adv}")

    lines += ["", "⚠️  Limitations"]
    for lim in candidate.get("limitations", []):
        lines.append(f"  • {lim}")

    lines += ["", "🔧 Preprocessing required"]
    for step in candidate.get("preprocessing_required", []):
        lines.append(f"  • {step}")

    lines += ["", "📈 Evaluation metrics"]
    lines.append(f"  {', '.join(candidate.get('metrics', []))}")

    lines += [
        "",
        f"💻 Computational cost: {candidate.get('computational_cost', 'unknown').upper()}",
        f"🔢 Minimum recommended samples: {candidate.get('min_samples', 'N/A')}",
        f"🔍 Handles missing values natively: {'Yes' if candidate.get('handles_missing') else 'No'}",
        f"🏷️  Handles categorical natively: {'Yes' if candidate.get('handles_categorical') else 'No'}",
        "",
        "🏁 Suitable when",
    ]
    for sw in candidate.get("suitable_when", []):
        lines.append(f"  • {sw}")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)
