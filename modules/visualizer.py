"""
Visualizer Module
==================
All Plotly visualization functions for the AutoML Model Selector.

Color scheme:
  primary   #2563EB  (blue)
  secondary #10B981  (green)
  neutral   #6B7280  (grey)
  accent    #F59E0B  (amber)
  danger    #EF4444  (red)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
PRIMARY = "#2563EB"
SECONDARY = "#10B981"
NEUTRAL = "#6B7280"
ACCENT = "#F59E0B"
DANGER = "#EF4444"

_PALETTE = [PRIMARY, SECONDARY, ACCENT, DANGER, "#8B5CF6", "#EC4899", "#14B8A6", "#F97316"]

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, Arial, sans-serif", size=13, color="#111827"),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#E5E7EB", borderwidth=1),
    xaxis=dict(showgrid=True, gridcolor="#F3F4F6", linecolor="#E5E7EB", zerolinecolor="#E5E7EB"),
    yaxis=dict(showgrid=True, gridcolor="#F3F4F6", linecolor="#E5E7EB", zerolinecolor="#E5E7EB"),
)


def _apply_theme(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply the standard theme to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#111827"), x=0.02),
        height=height,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def plot_missing_values(profile: Dict[str, Any]) -> go.Figure:
    """
    Bar chart of missing value percentages per column.

    Parameters
    ----------
    profile : dict
        Dataset profile containing ``missing_values`` (col → count) and ``n_rows``.

    Returns
    -------
    go.Figure
    """
    missing: Dict[str, int] = profile.get("missing_values", {})
    n_rows: int = profile.get("n_rows", 1) or 1

    if not missing:
        fig = go.Figure()
        fig.add_annotation(text="No missing values in dataset ✅", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=SECONDARY))
        return _apply_theme(fig, "Missing Values")

    cols = list(missing.keys())
    pcts = [cnt / n_rows * 100 for cnt in missing.values()]
    counts = list(missing.values())

    # Colour by severity
    colours = [DANGER if p > 50 else (ACCENT if p > 20 else PRIMARY) for p in pcts]

    fig = go.Figure(go.Bar(
        x=cols,
        y=pcts,
        marker_color=colours,
        text=[f"{p:.1f}%<br>({c:,})" for p, c in zip(pcts, counts)],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Missing: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_tickangle=-40,
        yaxis=dict(title="Missing %", range=[0, min(110, max(pcts) * 1.2)]),
    )
    return _apply_theme(fig, "Missing Values per Column")


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

def plot_distribution(
    df: pd.DataFrame,
    col: str,
    task_type: Optional[str] = None,
) -> go.Figure:
    """
    Histogram for numeric columns; bar chart for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column to visualise.
    task_type : str or None
        Informational – used to label the axis.

    Returns
    -------
    go.Figure
    """
    if col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{col}' not found.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return _apply_theme(fig, f"Distribution: {col}")

    series = df[col].dropna()

    if pd.api.types.is_numeric_dtype(series):
        fig = go.Figure(go.Histogram(
            x=series,
            nbinsx=min(50, max(10, len(series) // 20)),
            marker_color=PRIMARY,
            opacity=0.85,
            hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
        ))
        # Add KDE overlay if scipy available
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series.astype(float))
            x_range = np.linspace(series.min(), series.max(), 200)
            y_kde = kde(x_range) * len(series) * (series.max() - series.min()) / 40
            fig.add_trace(go.Scatter(
                x=x_range, y=y_kde,
                mode="lines", name="KDE",
                line=dict(color=SECONDARY, width=2),
            ))
        except Exception:
            pass
        fig.update_layout(xaxis_title=col, yaxis_title="Count")
    else:
        vc = series.value_counts().head(30)
        fig = go.Figure(go.Bar(
            x=vc.index.astype(str),
            y=vc.values,
            marker_color=PRIMARY,
            text=vc.values,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        ))
        fig.update_layout(xaxis_tickangle=-40, xaxis_title=col, yaxis_title="Count")

    return _apply_theme(fig, f"Distribution: {col}")


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    numerical_cols: List[str],
) -> go.Figure:
    """
    Plotly heatmap of the Pearson correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str

    Returns
    -------
    go.Figure
    """
    cols = [c for c in numerical_cols if c in df.columns]
    if len(cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need ≥ 2 numeric columns for correlation heatmap.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_theme(fig, "Correlation Heatmap")

    corr = df[cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y} × %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Pearson r"),
    ))
    h = max(400, 80 * len(cols))
    return _apply_theme(fig, "Feature Correlation Heatmap", height=h)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Optional[List[str]] = None,
) -> go.Figure:
    """
    Plotly heatmap-style confusion matrix.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    labels : list of str or None

    Returns
    -------
    go.Figure
    """
    try:
        from sklearn.metrics import confusion_matrix as sk_cm

        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        unique = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        if labels is None:
            labels = [str(u) for u in unique]

        cm = sk_cm(y_true_arr, y_pred_arr, labels=unique)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Build annotation text: count + percentage
        annotations = [
            [f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ]

        fig = go.Figure(go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            colorscale=[[0, "white"], [1, PRIMARY]],
            showscale=True,
            colorbar=dict(title="Rate"),
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z:.2%}<extra></extra>",
            text=annotations,
            texttemplate="%{text}",
        ))
        fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            xaxis=dict(side="bottom"),
        )
        h = max(350, 80 * len(labels))
        return _apply_theme(fig, "Confusion Matrix", height=h)

    except Exception as exc:
        logger.error("plot_confusion_matrix failed: %s", exc)
        fig = go.Figure()
        fig.add_annotation(text=f"Could not render confusion matrix: {exc}",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_theme(fig, "Confusion Matrix")


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: Sequence,
    y_pred_proba: np.ndarray,
    n_classes: int,
) -> go.Figure:
    """
    Plotly ROC curve. Supports binary and multi-class (OvR).

    Parameters
    ----------
    y_true : array-like
    y_pred_proba : np.ndarray, shape (n_samples,) or (n_samples, n_classes)
    n_classes : int

    Returns
    -------
    go.Figure
    """
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        y_true_arr = np.asarray(y_true)
        classes = np.unique(y_true_arr)
        fig = go.Figure()

        if n_classes == 2:
            proba = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
            fpr, tpr, _ = roc_curve(y_true_arr, proba)
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})",
                line=dict(color=PRIMARY, width=2),
            ))
        else:
            y_bin = label_binarize(y_true_arr, classes=classes)
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"Class {cls} (AUC={roc_auc:.3f})",
                    line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
                ))

        # Diagonal baseline
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color=NEUTRAL, width=1.5, dash="dash"),
        ))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.02]),
        )
        return _apply_theme(fig, "ROC Curve")

    except Exception as exc:
        logger.error("plot_roc_curve failed: %s", exc)
        fig = go.Figure()
        fig.add_annotation(text=f"ROC curve unavailable: {exc}",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_theme(fig, "ROC Curve")


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str,
    task_type: str,
) -> go.Figure:
    """
    Horizontal bar chart comparing models on a chosen metric.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Rows = models, columns include the metric.
    metric : str
        Metric column name.
    task_type : str

    Returns
    -------
    go.Figure
    """
    if metric not in comparison_df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Metric '{metric}' not found in results.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_theme(fig, "Model Comparison")

    df_sorted = comparison_df.sort_values(metric, ascending=True).tail(20)
    model_names = df_sorted.index.astype(str).tolist()
    values = df_sorted[metric].tolist()

    # Colour best bar
    max_val = max(values) if values else 0
    colours = [SECONDARY if v == max_val else PRIMARY for v in values]

    fig = go.Figure(go.Bar(
        y=model_names,
        x=values,
        orientation="h",
        marker_color=colours,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>" + metric + ": %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=metric,
        yaxis_title="Model",
        margin=dict(l=160, r=40, t=60, b=50),
    )
    h = max(350, 45 * len(model_names))
    return _apply_theme(fig, f"Model Comparison – {metric}", height=h)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """
    Horizontal bar chart of top-N feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must have ``feature`` and ``importance`` columns.
    top_n : int

    Returns
    -------
    go.Figure
    """
    if importance_df is None or importance_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No feature importance data available.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_theme(fig, "Feature Importance")

    df = importance_df.head(top_n).copy()
    # Sort ascending for horizontal bar (top feature at top)
    df = df.sort_values("importance", ascending=True)

    has_std = "std" in df.columns

    fig = go.Figure(go.Bar(
        y=df["feature"],
        x=df["importance"],
        orientation="h",
        marker_color=PRIMARY,
        error_x=dict(type="data", array=df["std"].tolist()) if has_std else None,
        text=[f"{v:.4f}" for v in df["importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        margin=dict(l=160, r=60, t=60, b=50),
    )
    h = max(350, 30 * len(df))
    return _apply_theme(fig, f"Top {min(top_n, len(df))} Feature Importances", height=h)


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: Sequence,
    y_pred: Sequence,
) -> go.Figure:
    """
    Residual scatter plot (predicted vs. residual).

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    go.Figure
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred_arr, y=residuals, mode="markers",
        marker=dict(color=PRIMARY, opacity=0.6, size=6),
        name="Residuals",
        hovertemplate="Predicted: %{x:.4f}<br>Residual: %{y:.4f}<extra></extra>",
    ))
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=DANGER, line_width=1.5)
    fig.update_layout(
        xaxis_title="Predicted Value",
        yaxis_title="Residual (Actual − Predicted)",
    )
    return _apply_theme(fig, "Residual Plot")


# ---------------------------------------------------------------------------
# Actual vs predicted
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true: Sequence,
    y_pred: Sequence,
) -> go.Figure:
    """
    Scatter plot of actual vs. predicted values with ideal line.

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    go.Figure
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true_arr, y=y_pred_arr, mode="markers",
        marker=dict(color=PRIMARY, opacity=0.55, size=6),
        name="Predictions",
        hovertemplate="Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>",
    ))
    # Perfect prediction line
    mn = min(y_true_arr.min(), y_pred_arr.min())
    mx = max(y_true_arr.max(), y_pred_arr.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines", name="Perfect Fit",
        line=dict(color=SECONDARY, width=2, dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Actual Value",
        yaxis_title="Predicted Value",
    )
    return _apply_theme(fig, "Actual vs. Predicted")


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    y: Sequence,
    title: str = "Class Distribution",
) -> go.Figure:
    """
    Pie chart (≤ 6 classes) or bar chart of target class distribution.

    Parameters
    ----------
    y : array-like
        Target labels.
    title : str

    Returns
    -------
    go.Figure
    """
    series = pd.Series(y).value_counts().sort_index()
    labels = series.index.astype(str).tolist()
    values = series.values.tolist()

    if len(labels) <= 6:
        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.35,
            marker_colors=_PALETTE[:len(labels)],
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
    else:
        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=PRIMARY,
            text=values,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        ))
        fig.update_layout(xaxis_tickangle=-40, xaxis_title="Class", yaxis_title="Count")

    return _apply_theme(fig, title)


# ---------------------------------------------------------------------------
# Outlier box plots
# ---------------------------------------------------------------------------

def plot_outlier_boxplots(
    df: pd.DataFrame,
    numerical_cols: List[str],
    max_cols: int = 6,
) -> go.Figure:
    """
    Box plots for up to ``max_cols`` numeric columns to visualise outliers.

    Parameters
    ----------
    df : pd.DataFrame
    numerical_cols : list of str
    max_cols : int

    Returns
    -------
    go.Figure
    """
    cols = [c for c in numerical_cols if c in df.columns][:max_cols]
    if not cols:
        fig = go.Figure()
        fig.add_annotation(text="No numerical columns to plot.", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return _apply_theme(fig, "Outlier Box Plots")

    n = len(cols)
    ncols_grid = min(3, n)
    nrows_grid = (n + ncols_grid - 1) // ncols_grid

    fig = make_subplots(rows=nrows_grid, cols=ncols_grid, subplot_titles=cols)

    for i, col in enumerate(cols):
        row = i // ncols_grid + 1
        col_idx = i % ncols_grid + 1
        series = df[col].dropna()
        fig.add_trace(
            go.Box(
                y=series,
                name=col,
                marker_color=_PALETTE[i % len(_PALETTE)],
                boxmean="sd",
                hovertemplate=f"<b>{col}</b><br>" + "Value: %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col_idx,
        )

    h = max(350, 250 * nrows_grid)
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=h,
        title=dict(text="Outlier Box Plots", font=dict(size=15, color="#111827"), x=0.02),
        font=dict(family="Inter, Arial, sans-serif", size=12),
        margin=dict(l=50, r=30, t=80, b=50),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#F3F4F6")
    return fig


# ---------------------------------------------------------------------------
# Pair plot (scatter matrix)
# ---------------------------------------------------------------------------

def plot_pairplot(
    df: pd.DataFrame,
    target_col: str,
    max_features: int = 5,
) -> go.Figure:
    """
    Plotly scatter matrix (pair plot) coloured by target column.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Used for colouring points.
    max_features : int
        Maximum number of numeric features to include.

    Returns
    -------
    go.Figure
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    num_cols = num_cols[:max_features]

    if len(num_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need ≥ 2 numeric features for pair plot.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_theme(fig, "Pair Plot")

    plot_df = df[num_cols + ([target_col] if target_col in df.columns else [])].dropna()

    color_col = target_col if target_col in plot_df.columns else None
    try:
        fig = px.scatter_matrix(
            plot_df,
            dimensions=num_cols,
            color=color_col,
            color_discrete_sequence=_PALETTE,
            opacity=0.55,
            title="Pair Plot",
            labels={c: c for c in num_cols},
        )
        fig.update_traces(diagonal_visible=False, marker=dict(size=4))
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=600,
            font=dict(family="Inter, Arial, sans-serif", size=11),
        )
    except Exception as exc:
        logger.error("plot_pairplot failed: %s", exc)
        fig = go.Figure()
        fig.add_annotation(text=f"Pair plot unavailable: {exc}",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_theme(fig, "Pair Plot")

    return fig


# ---------------------------------------------------------------------------
# Cluster scatter
# ---------------------------------------------------------------------------

def plot_cluster_scatter(
    X_2d: np.ndarray,
    labels: Sequence,
    title: str = "Clusters",
) -> go.Figure:
    """
    2-D scatter plot for clustering results (e.g. after PCA/TSNE reduction).

    Parameters
    ----------
    X_2d : np.ndarray, shape (n_samples, 2)
        2-D embedding of samples.
    labels : array-like
        Cluster labels for each sample.
    title : str

    Returns
    -------
    go.Figure
    """
    X_arr = np.asarray(X_2d)
    labels_arr = np.asarray(labels)
    unique_labels = np.unique(labels_arr)

    fig = go.Figure()
    for i, cluster_id in enumerate(unique_labels):
        mask = labels_arr == cluster_id
        label_name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        colour = NEUTRAL if cluster_id == -1 else _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=X_arr[mask, 0],
            y=X_arr[mask, 1],
            mode="markers",
            name=label_name,
            marker=dict(color=colour, size=6, opacity=0.7),
            hovertemplate=f"<b>{label_name}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Component 1",
        yaxis_title="Component 2",
    )
    return _apply_theme(fig, title)


# ---------------------------------------------------------------------------
# Training time comparison
# ---------------------------------------------------------------------------

def plot_training_time_comparison(results_list: List[Dict[str, Any]]) -> go.Figure:
    """
    Bar chart comparing training times across models.

    Parameters
    ----------
    results_list : list of dict
        Each dict must have ``model_name`` and ``training_time`` keys.

    Returns
    -------
    go.Figure
    """
    valid = [r for r in results_list if r.get("training_time") is not None]
    if not valid:
        fig = go.Figure()
        fig.add_annotation(text="No training time data available.",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return _apply_theme(fig, "Training Time Comparison")

    sorted_results = sorted(valid, key=lambda r: r["training_time"])
    names = [r.get("model_name", "Unknown") for r in sorted_results]
    times = [r["training_time"] for r in sorted_results]

    min_time = min(times)
    colours = [SECONDARY if t == min_time else PRIMARY for t in times]

    fig = go.Figure(go.Bar(
        y=names,
        x=times,
        orientation="h",
        marker_color=colours,
        text=[f"{t:.2f}s" for t in times],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Training time: %{x:.3f}s<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Training Time (seconds)",
        yaxis_title="Model",
        margin=dict(l=160, r=80, t=60, b=50),
    )
    h = max(350, 45 * len(names))
    return _apply_theme(fig, "Training Time Comparison", height=h)
