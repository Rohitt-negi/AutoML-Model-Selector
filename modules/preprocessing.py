"""
preprocessing.py
================
Preprocessing pipeline builder for the AutoML Model Selector.

Provides utilities to build sklearn-compatible preprocessing pipelines,
split datasets, encode targets, retrieve post-transformation feature names,
detect potential data leakage, and identify free-text columns.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    StandardScaler,
)

logger = logging.getLogger(__name__)

# Threshold for "high cardinality" categoricals — above this we use OrdinalEncoder
_HIGH_CARDINALITY_THRESHOLD = 50

# Fraction of unique values relative to column length above which a column is
# suspected of being free text rather than a categorical variable.
_TEXT_UNIQUENESS_THRESHOLD = 0.8

# Average token count above which a column is considered free text.
_TEXT_AVG_TOKENS_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_numerical_transformer(strategy: str = "median") -> Pipeline:
    """Return a Pipeline that imputes then scales numerical features.

    Parameters
    ----------
    strategy : str
        Imputation strategy passed to :class:`SimpleImputer`.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=strategy)),
            ("scaler", StandardScaler()),
        ]
    )


def _build_categorical_transformer(
    unique_values: int,
    handle_unknown: str = "ignore",
) -> Pipeline:
    """Return a Pipeline for categorical features.

    Uses ``OneHotEncoder`` for low-cardinality columns and ``OrdinalEncoder``
    for high-cardinality ones (> _HIGH_CARDINALITY_THRESHOLD unique values).

    Parameters
    ----------
    unique_values : int
        Number of unique values in the column (used to decide encoder type).
    handle_unknown : str
        Strategy for unknown categories at inference time.
    """
    if unique_values > _HIGH_CARDINALITY_THRESHOLD:
        # High-cardinality: use OrdinalEncoder to avoid exploding dimensionality
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    else:
        from sklearn.preprocessing import OneHotEncoder  # local import for clarity

        encoder = OneHotEncoder(
            handle_unknown=handle_unknown,
            sparse_output=False,
        )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            ("encoder", encoder),
        ]
    )


def _drop_all_missing_columns(df: pd.DataFrame, threshold: float = 1.0) -> list[str]:
    """Return column names where the fraction of missing values equals ``threshold``."""
    frac_missing = df.isnull().mean()
    return list(frac_missing[frac_missing >= threshold].index)


def _drop_constant_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that contain only a single unique non-null value."""
    return [col for col in df.columns if df[col].nunique(dropna=True) <= 1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Build and return a :class:`~sklearn.compose.ColumnTransformer` preprocessing pipeline.

    The transformer handles:
    - Missing value imputation (median for numericals, most_frequent for categoricals)
    - Standard scaling for numerical features
    - OneHotEncoder for low-cardinality categoricals, OrdinalEncoder for high-cardinality ones
    - Drops all-missing and constant columns before building transformers

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe (including target column).
    target_col : str
        Name of the target column.
    task_type : str
        Task type string (used for future-proofing; not currently altering logic).
    numerical_cols : list[str]
        List of numerical column names (must not include target).
    categorical_cols : list[str]
        List of categorical column names (must not include target).

    Returns
    -------
    ColumnTransformer
        Unfitted sklearn ColumnTransformer ready to be placed in a Pipeline.

    Raises
    ------
    ValueError
        If both ``numerical_cols`` and ``categorical_cols`` are empty.
    """
    if not numerical_cols and not categorical_cols:
        raise ValueError(
            "Both numerical_cols and categorical_cols are empty. "
            "Provide at least one feature column."
        )

    feature_df = df.drop(columns=[target_col], errors="ignore")

    # Identify and warn about all-missing columns
    all_missing = set(_drop_all_missing_columns(feature_df))
    if all_missing:
        logger.warning(
            "Columns with 100%% missing values will be dropped: %s",
            sorted(all_missing),
        )

    # Identify and warn about constant columns
    constants = set(_drop_constant_columns(feature_df))
    if constants:
        logger.warning(
            "Constant columns (single unique value) will be dropped: %s",
            sorted(constants),
        )

    drop_set = all_missing | constants

    # Filter feature lists
    numerical_cols = [c for c in numerical_cols if c not in drop_set and c in feature_df.columns]
    categorical_cols = [c for c in categorical_cols if c not in drop_set and c in feature_df.columns]

    transformers: list[tuple] = []

    # Numerical transformer
    if numerical_cols:
        num_transformer = _build_numerical_transformer(strategy="median")
        transformers.append(("numerical", num_transformer, numerical_cols))

    # Categorical transformers — per column to allow per-column cardinality check
    for col in categorical_cols:
        n_unique = df[col].nunique(dropna=True)
        cat_transformer = _build_categorical_transformer(unique_values=n_unique)
        transformers.append((f"cat_{col}", cat_transformer, [col]))

    if not transformers:
        raise ValueError(
            "No valid feature columns remain after dropping all-missing and constant columns."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop any columns not explicitly listed
        verbose_feature_names_out=True,
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a dataframe into training and test sets.

    Uses stratified splitting for classification tasks to preserve class distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target column.
    target_col : str
        Name of the target column.
    task_type : str
        Task type; stratification is applied for ``'classification'`` and ``'nlp'``.
    test_size : float, optional
        Fraction of data to reserve for testing (default 0.2).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``(X_train, X_test, y_train, y_test)``

    Raises
    ------
    ValueError
        If ``target_col`` is not present in ``df``.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = None
    if task_type.lower() in ("classification", "nlp"):
        # Only stratify when each class has at least 2 samples
        class_counts = y.value_counts()
        if class_counts.min() >= 2:
            stratify = y
        else:
            warnings.warn(
                "Some classes have fewer than 2 samples — stratification disabled.",
                UserWarning,
                stacklevel=2,
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info(
        "Split: %d train / %d test (test_size=%.0f%%)",
        len(X_train),
        len(X_test),
        test_size * 100,
    )

    return X_train, X_test, y_train, y_test


def encode_target(
    y: pd.Series,
    task_type: str,
) -> tuple[np.ndarray, LabelEncoder | None]:
    """Encode the target variable for the given task type.

    For classification tasks the target is label-encoded to contiguous integers.
    For other tasks the target is returned as a numpy array without transformation.

    Parameters
    ----------
    y : pd.Series
        Target series.
    task_type : str
        Task type string.

    Returns
    -------
    tuple[np.ndarray, LabelEncoder | None]
        ``(encoded_y, encoder)`` where ``encoder`` is ``None`` for non-classification tasks.
    """
    if task_type.lower() in ("classification", "nlp"):
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        return y_encoded, encoder

    # Regression, clustering, time_series, anomaly_detection — no target encoding
    return np.asarray(y), None


def get_feature_names(
    preprocessor: ColumnTransformer,
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    """Retrieve feature names after ColumnTransformer transformation.

    Provides a graceful fallback that generates generic names if
    ``get_feature_names_out`` is not available or raises an error.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A **fitted** ColumnTransformer.
    numerical_cols : list[str]
        Numerical column names originally passed to the transformer.
    categorical_cols : list[str]
        Categorical column names originally passed to the transformer.

    Returns
    -------
    list[str]
        List of output feature name strings.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        pass

    # Fallback: reconstruct names manually
    feature_names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if isinstance(cols, list):
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    sub_names = transformer.get_feature_names_out(cols)
                    feature_names.extend(sub_names)
                    continue
                except Exception:
                    pass
            feature_names.extend([f"{name}__{c}" for c in cols])
        else:
            feature_names.append(str(cols))

    return feature_names


def check_data_leakage_risks(
    df: pd.DataFrame,
    target_col: str,
) -> list[str]:
    """Scan the dataframe for potential data leakage risks.

    Checks for:
    - Columns with near-perfect correlation to the target
    - ID-like columns (integer sequence or UUID patterns)
    - Timestamp columns that may encode future information
    - Columns that are a deterministic function of the target

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including target column.
    target_col : str
        Name of the target column.

    Returns
    -------
    list[str]
        List of human-readable warning strings.  Empty list means no risks found.
    """
    warnings_list: list[str] = []

    if target_col not in df.columns:
        return [f"Target column '{target_col}' not found — cannot check leakage."]

    feature_cols = [c for c in df.columns if c != target_col]

    # 1. Near-perfect correlation with target (numerical features only)
    target_numeric = pd.to_numeric(df[target_col], errors="coerce")
    if target_numeric.notna().sum() > 1:
        for col in feature_cols:
            col_numeric = pd.to_numeric(df[col], errors="coerce")
            if col_numeric.notna().sum() < 2:
                continue
            try:
                correlation = abs(col_numeric.corr(target_numeric))
                if correlation > 0.95:
                    warnings_list.append(
                        f"⚠️  Column '{col}' has very high correlation ({correlation:.3f}) "
                        f"with target '{target_col}'. Possible leakage."
                    )
            except Exception:
                pass

    # 2. ID-like columns (monotonically increasing integer or unique string)
    for col in feature_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Check for integer sequence
        if pd.api.types.is_integer_dtype(series):
            diffs = series.diff().dropna()
            if (diffs == 1).all():
                warnings_list.append(
                    f"⚠️  Column '{col}' appears to be a sequential ID. "
                    "Consider excluding it from training."
                )
                continue

        # Check for high uniqueness ratio (likely ID or UUID)
        uniqueness = series.nunique() / len(series)
        if uniqueness > 0.95 and series.dtype == object:
            warnings_list.append(
                f"⚠️  Column '{col}' has {uniqueness:.0%} unique values. "
                "Likely an ID or UUID — consider dropping."
            )

    # 3. Datetime / timestamp columns
    for col in feature_cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            warnings_list.append(
                f"⚠️  Column '{col}' is a datetime. Ensure it does not encode future "
                "information relative to the target (temporal leakage)."
            )
        elif df[col].dtype == object:
            # Try parsing as datetime
            sample = df[col].dropna().head(20)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    warnings_list.append(
                        f"⚠️  Column '{col}' may be a date/time string. "
                        "Check for temporal leakage."
                    )
            except Exception:
                pass

    # 4. Target-derived columns (e.g., ratio, log of target that was accidentally left in)
    if pd.api.types.is_numeric_dtype(target_numeric):
        for col in feature_cols:
            col_numeric = pd.to_numeric(df[col], errors="coerce").dropna()
            target_clean = target_numeric.dropna()
            common_idx = col_numeric.index.intersection(target_clean.index)
            if len(common_idx) < 2:
                continue
            try:
                ratio = col_numeric.loc[common_idx] / (target_clean.loc[common_idx] + 1e-9)
                if ratio.std() < 1e-6 and ratio.notna().mean() > 0.95:
                    warnings_list.append(
                        f"⚠️  Column '{col}' may be a near-constant multiple of the target. "
                        "Possible target leakage."
                    )
            except Exception:
                pass

    if not warnings_list:
        warnings_list.append("✅ No obvious data leakage risks detected.")

    return warnings_list


def detect_text_columns(df: pd.DataFrame) -> list[str]:
    """Identify columns in ``df`` that are likely free-text rather than categorical.

    Heuristics:
    - Column dtype is ``object``
    - High uniqueness ratio (> ``_TEXT_UNIQUENESS_THRESHOLD``)  OR
      average number of whitespace-separated tokens > ``_TEXT_AVG_TOKENS_THRESHOLD``

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    list[str]
        Column names suspected of being free-text fields.
    """
    text_cols: list[str] = []

    for col in df.select_dtypes(include="object").columns:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        uniqueness_ratio = series.nunique() / len(series)
        avg_tokens = series.str.split().str.len().mean()

        if uniqueness_ratio > _TEXT_UNIQUENESS_THRESHOLD or avg_tokens > _TEXT_AVG_TOKENS_THRESHOLD:
            text_cols.append(col)

    return text_cols
