"""
Report Generation & Export Module
====================================
Generates Markdown reports, CSV exports, PDF summaries, and serialised model
artifacts suitable for Streamlit ``st.download_button`` calls.
"""

from __future__ import annotations

import io
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDF availability guard
# ---------------------------------------------------------------------------
try:
    from fpdf import FPDF

    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False
    logger.info("fpdf2 not installed – PDF export will fall back to Markdown.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _safe_float(val: Any, decimals: int = 4) -> str:
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _metric_table_md(metrics: Dict[str, Any]) -> str:
    lines = ["| Metric | Value |", "|--------|-------|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {_safe_float(v)} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset report
# ---------------------------------------------------------------------------

def generate_dataset_report(profile: Dict[str, Any], df: pd.DataFrame) -> str:
    """
    Generate a Markdown report describing the dataset.

    Parameters
    ----------
    profile : dict
        Output of a dataset profiling step (keys like ``n_rows``, ``n_cols``,
        ``missing_values``, ``column_types``, ``target_column``, etc.).
    df : pd.DataFrame
        The raw dataset.

    Returns
    -------
    str
        Markdown-formatted report string.
    """
    lines: List[str] = []
    ts = _now_str()

    lines.append(f"# Dataset Analysis Report")
    lines.append(f"\n_Generated: {ts}_\n")

    # --- Overview ---
    n_rows = profile.get("n_rows", len(df))
    n_cols = profile.get("n_cols", len(df.columns))
    target = profile.get("target_column", "N/A")
    task = profile.get("task_type", "N/A")

    lines.append("## Overview")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Rows | {n_rows:,} |")
    lines.append(f"| Columns | {n_cols} |")
    lines.append(f"| Target Column | `{target}` |")
    lines.append(f"| Task Type | {task} |")

    # Memory usage
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    lines.append(f"| Memory Usage | {mem_mb:.2f} MB |")
    lines.append("")

    # --- Missing values ---
    lines.append("## Missing Values")
    missing = profile.get("missing_values", {})
    if not missing:
        missing = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}

    if missing:
        lines.append("| Column | Missing Count | Missing % |")
        lines.append("|--------|--------------|-----------|")
        for col, cnt in sorted(missing.items(), key=lambda x: x[1], reverse=True):
            pct = cnt / n_rows * 100 if n_rows else 0
            lines.append(f"| `{col}` | {cnt:,} | {pct:.1f}% |")
    else:
        lines.append("✅ No missing values detected.")
    lines.append("")

    # --- Column types ---
    lines.append("## Column Types")
    col_types = profile.get("column_types", {})
    type_counts: Dict[str, int] = {}
    if isinstance(col_types, dict):
        for k, v in col_types.items():
            if isinstance(v, list):
                type_counts[k.capitalize()] = len(v)
            else:
                type_counts[str(v)] = type_counts.get(str(v), 0) + 1
    else:
        for col in df.columns:
            dt = str(df[col].dtype)
            type_counts[dt] = type_counts.get(dt, 0) + 1

    lines.append("| Category / Type | Count |")
    lines.append("|-----------------|-------|")
    for dtype, cnt in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {dtype} | {cnt} |")
    lines.append("")

    # --- Numeric summary ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        lines.append("## Numeric Column Statistics")
        try:
            desc = df[num_cols].describe().T.reset_index()
            desc.rename(columns={"index": "column"}, inplace=True)
            lines.append(desc.to_markdown(index=False, floatfmt=".4f"))
        except Exception:
            lines.append("_(Statistics unavailable)_")
        lines.append("")

    # --- Categorical summary ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        lines.append("## Categorical Column Summary")
        lines.append("| Column | Unique Values | Top Value | Frequency |")
        lines.append("|--------|--------------|-----------|-----------|")
        for col in cat_cols[:20]:  # limit
            vc = df[col].value_counts()
            top_val = vc.index[0] if not vc.empty else "N/A"
            top_freq = vc.iloc[0] if not vc.empty else 0
            lines.append(f"| `{col}` | {df[col].nunique()} | {top_val} | {top_freq:,} |")
        lines.append("")

    # --- Duplicate rows ---
    n_dup = int(df.duplicated().sum())
    lines.append("## Data Quality Checks")
    lines.append(f"- **Duplicate rows:** {n_dup:,} ({n_dup/n_rows*100:.1f}%)" if n_rows else f"- **Duplicate rows:** {n_dup}")
    lines.append(f"- **Constant columns:** {sum(1 for c in df.columns if df[c].nunique() <= 1)}")
    lines.append(f"- **Columns with >50% missing:** {sum(1 for c in df.columns if df[c].isna().mean() > 0.5)}")
    lines.append("")

    # --- Outliers (IQR) ---
    if num_cols:
        outlier_info = []
        for col in num_cols[:10]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            if n_out > 0:
                outlier_info.append(f"| `{col}` | {n_out:,} ({n_out/n_rows*100:.1f}%) |")
        if outlier_info:
            lines.append("## Outlier Summary (IQR Method)")
            lines.append("| Column | Outlier Count |")
            lines.append("|--------|--------------|")
            lines.extend(outlier_info)
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model comparison CSV
# ---------------------------------------------------------------------------

def generate_model_comparison_csv(comparison_df: pd.DataFrame) -> bytes:
    """
    Serialise a model comparison DataFrame to CSV bytes.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame where each row is a model and columns are metrics.

    Returns
    -------
    bytes
        UTF-8 encoded CSV suitable for ``st.download_button``.
    """
    buf = io.StringIO()
    comparison_df.to_csv(buf, index=True)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# PDF summary
# ---------------------------------------------------------------------------

def generate_summary_pdf(
    profile: Dict[str, Any],
    results_list: List[Dict[str, Any]],
    best_result: Dict[str, Any],
    task_type: str,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Generate a PDF summary report using fpdf2.

    Falls back to UTF-8 Markdown bytes if fpdf2 is unavailable.

    Parameters
    ----------
    profile : dict
        Dataset profile dict.
    results_list : list of dict
        All model results (each dict has ``model_name``, ``metrics``, etc.).
    best_result : dict
        The single best result dict.
    task_type : str
        ``"classification"`` or ``"regression"``.
    output_path : str or None
        Optional file path to save the PDF to disk. If None, returns bytes only.

    Returns
    -------
    bytes
        Raw PDF bytes or Markdown bytes if PDF unavailable.
    """
    if not _FPDF_AVAILABLE:
        logger.warning("fpdf2 not available – generating Markdown fallback.")
        md = _generate_summary_markdown(profile, results_list, best_result, task_type)
        return md.encode("utf-8")

    try:
        return _build_pdf(profile, results_list, best_result, task_type, output_path)
    except Exception as exc:
        logger.error("PDF generation failed: %s\n%s", exc, traceback.format_exc())
        md = _generate_summary_markdown(profile, results_list, best_result, task_type)
        return md.encode("utf-8")


def _generate_summary_markdown(
    profile: Dict[str, Any],
    results_list: List[Dict[str, Any]],
    best_result: Dict[str, Any],
    task_type: str,
) -> str:
    lines = [
        "# AutoML Model Selector – Summary Report",
        f"\n_Generated: {_now_str()}_\n",
        "## Dataset Summary",
        f"- **Rows:** {profile.get('n_rows', 'N/A'):,}" if isinstance(profile.get('n_rows'), int) else f"- **Rows:** {profile.get('n_rows', 'N/A')}",
        f"- **Columns:** {profile.get('n_cols', 'N/A')}",
        f"- **Target:** `{profile.get('target_column', 'N/A')}`",
        f"- **Task Type:** {task_type}",
        "",
        "## Model Comparison",
        "| Model | " + " | ".join(best_result.get("metrics", {}).keys()) + " |",
        "|-------|" + "|".join(["------"] * len(best_result.get("metrics", {}))) + "|",
    ]
    for r in results_list:
        metrics = r.get("metrics", {})
        row = f"| {r.get('model_name', 'Unknown')} | " + " | ".join(_safe_float(v) for v in metrics.values()) + " |"
        lines.append(row)

    lines += [
        "",
        "## Best Model",
        f"**{best_result.get('model_name', 'N/A')}**",
        "",
        _metric_table_md(best_result.get("metrics", {})),
    ]
    return "\n".join(lines)


def _build_pdf(
    profile: Dict[str, Any],
    results_list: List[Dict[str, Any]],
    best_result: Dict[str, Any],
    task_type: str,
    output_path: Optional[str],
) -> bytes:
    """Internal fpdf2-based PDF builder."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "AutoML Model Selector", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Summary Report", ln=True, align="C")
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, f"Generated: {_now_str()}", ln=True, align="C")
    pdf.ln(8)

    # Dataset summary box
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Dataset Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)

    summary_items = [
        ("Rows", f"{profile.get('n_rows', 'N/A'):,}" if isinstance(profile.get("n_rows"), int) else str(profile.get("n_rows", "N/A"))),
        ("Columns", str(profile.get("n_cols", "N/A"))),
        ("Target Column", str(profile.get("target_column", "N/A"))),
        ("Task Type", str(task_type)),
        ("Missing Values", str(len(profile.get("missing_values", {}))) + " column(s) affected"),
    ]
    col_w = [60, 120]
    for label, value in summary_items:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(col_w[0], 7, label, border=0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(col_w[1], 7, value, border=0, ln=True)
    pdf.ln(6)

    # Model comparison table
    if results_list:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Model Comparison", ln=True)
        pdf.ln(2)

        # Gather all metric keys
        metric_keys = list(results_list[0].get("metrics", {}).keys())
        headers = ["Model"] + metric_keys
        col_widths = _compute_col_widths(headers, pdf)

        # Header row
        pdf.set_fill_color(37, 99, 235)  # #2563EB
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        for header, w in zip(headers, col_widths):
            pdf.cell(w, 7, header[:18], border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_text_color(0, 0, 0)
        best_name = best_result.get("model_name", "")
        for i, r in enumerate(results_list):
            fill = i % 2 == 0
            pdf.set_fill_color(240, 245, 255) if fill else pdf.set_fill_color(255, 255, 255)
            name = r.get("model_name", "Unknown")
            if name == best_name:
                pdf.set_font("Helvetica", "B", 9)
            else:
                pdf.set_font("Helvetica", "", 9)
            pdf.cell(col_widths[0], 6, name[:22], border=1, fill=fill)
            for key, w in zip(metric_keys, col_widths[1:]):
                val = _safe_float(r.get("metrics", {}).get(key, "N/A"))
                pdf.cell(w, 6, val, border=1, align="R", fill=fill)
            pdf.ln()
        pdf.ln(8)

    # Best model section
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Best Model", ln=True)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(16, 185, 129)  # #10B981
    pdf.cell(0, 8, best_result.get("model_name", "N/A"), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    for k, v in best_result.get("metrics", {}).items():
        pdf.cell(80, 6, k, border="B")
        pdf.cell(60, 6, _safe_float(v), border="B", align="R", ln=True)
    pdf.ln(6)

    # Training time
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Training Time", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for r in results_list:
        t = r.get("training_time", None)
        if t is not None:
            pdf.cell(0, 6, f"{r.get('model_name', 'Unknown')}: {t:.2f}s", ln=True)
    pdf.ln(4)

    # Footer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(107, 114, 128)
    pdf.cell(0, 6, "AutoML Model Selector | Confidential", align="C", ln=True)

    pdf_bytes = pdf.output()
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info("PDF saved to %s", output_path)

    return bytes(pdf_bytes)


def _compute_col_widths(headers: List[str], pdf: Any) -> List[float]:
    """Distribute page width among columns proportionally."""
    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    # Give model name column a larger share
    if len(headers) <= 1:
        return [page_width]
    name_w = min(50, page_width * 0.35)
    remainder = page_width - name_w
    metric_w = remainder / max(len(headers) - 1, 1)
    return [name_w] + [metric_w] * (len(headers) - 1)


# ---------------------------------------------------------------------------
# Model artifact export
# ---------------------------------------------------------------------------

def export_model_artifact(
    pipeline: Any,
    model_id: str,
    metadata: Dict[str, Any],
) -> bytes:
    """
    Serialise a fitted pipeline + metadata using joblib and return raw bytes.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline to export.
    model_id : str
        Unique identifier / name for this model.
    metadata : dict
        Arbitrary metadata to bundle with the artifact
        (e.g. metrics, feature names, task type, training timestamp).

    Returns
    -------
    bytes
        Joblib-serialised bytes of ``{"model_id": ..., "pipeline": ..., "metadata": ...}``.
    """
    artifact = {
        "model_id": model_id,
        "pipeline": pipeline,
        "metadata": {
            **metadata,
            "export_timestamp": _now_str(),
        },
    }
    buf = io.BytesIO()
    joblib.dump(artifact, buf, compress=3)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Comprehensive evaluation report
# ---------------------------------------------------------------------------

def generate_evaluation_report(
    results_list: List[Dict[str, Any]],
    task_type: str,
    profile: Dict[str, Any],
) -> str:
    """
    Generate a comprehensive Markdown evaluation report covering all trained models.

    Parameters
    ----------
    results_list : list of dict
        Each dict must have at minimum: ``model_name``, ``metrics``,
        ``training_time`` (optional), ``cv_scores`` (optional).
    task_type : str
        ``"classification"`` or ``"regression"``.
    profile : dict
        Dataset profile.

    Returns
    -------
    str
        Full Markdown evaluation report.
    """
    lines: List[str] = []
    ts = _now_str()

    lines.append("# AutoML Evaluation Report")
    lines.append(f"\n_Generated: {ts}_\n")

    # Dataset context
    lines.append("## Dataset Context")
    lines.append(f"- **Rows:** {profile.get('n_rows', 'N/A')}")
    lines.append(f"- **Features:** {profile.get('n_cols', 'N/A')}")
    lines.append(f"- **Target:** `{profile.get('target_column', 'N/A')}`")
    lines.append(f"- **Task Type:** {task_type}")
    lines.append("")

    if not results_list:
        lines.append("_No results available._")
        return "\n".join(lines)

    # Determine primary metric
    primary_metric = _primary_metric(task_type)

    # Sort results by primary metric
    def _sort_key(r: Dict[str, Any]) -> float:
        val = r.get("metrics", {}).get(primary_metric, None)
        if val is None:
            return -float("inf") if task_type != "regression" else float("inf")
        return float(val)

    ascending = task_type == "regression" and primary_metric in ("rmse", "mae", "mse")
    sorted_results = sorted(results_list, key=_sort_key, reverse=not ascending)
    best = sorted_results[0]

    # Executive summary
    lines.append("## Executive Summary")
    lines.append(
        f"**{len(results_list)} models** were evaluated. "
        f"The best performing model is **{best.get('model_name', 'N/A')}** "
        f"with a {primary_metric} of **{_safe_float(best.get('metrics', {}).get(primary_metric))}**."
    )
    lines.append("")

    # Comparison table
    metric_keys = list(sorted_results[0].get("metrics", {}).keys())
    lines.append("## Model Comparison Table")
    header = "| Rank | Model | " + " | ".join(metric_keys) + " | Training Time |"
    sep = "|------|-------|" + "|".join(["------"] * len(metric_keys)) + "|---------------|"
    lines.append(header)
    lines.append(sep)
    for rank, r in enumerate(sorted_results, start=1):
        name = r.get("model_name", "Unknown")
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else ""))
        metrics_str = " | ".join(_safe_float(r.get("metrics", {}).get(k)) for k in metric_keys)
        t = r.get("training_time")
        t_str = f"{t:.2f}s" if t is not None else "N/A"
        lines.append(f"| {rank} {medal} | {name} | {metrics_str} | {t_str} |")
    lines.append("")

    # Per-model details
    lines.append("## Detailed Model Reports")
    for r in sorted_results:
        name = r.get("model_name", "Unknown")
        lines.append(f"### {name}")

        # Metrics
        lines.append("**Metrics:**")
        lines.append(_metric_table_md(r.get("metrics", {})))
        lines.append("")

        # CV scores
        cv = r.get("cv_scores")
        if cv is not None and len(cv) > 0:
            mean_cv = np.mean(cv)
            std_cv = np.std(cv)
            lines.append(f"**Cross-Validation ({len(cv)} folds):**")
            lines.append(f"- Mean: {mean_cv:.4f}")
            lines.append(f"- Std:  {std_cv:.4f}")
            lines.append(f"- Scores: {', '.join(f'{s:.4f}' for s in cv)}")
            lines.append("")

        # Hyperparameters
        params = r.get("best_params") or r.get("params")
        if params:
            lines.append("**Hyperparameters:**")
            lines.append("```")
            for k, v in params.items():
                lines.append(f"  {k}: {v}")
            lines.append("```")
            lines.append("")

        # Training time
        t = r.get("training_time")
        if t is not None:
            lines.append(f"**Training time:** {t:.2f}s")
            lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append(f"1. **Deploy {best.get('model_name', 'N/A')}** as the primary model.")
    if len(sorted_results) > 1:
        runner_up = sorted_results[1]
        lines.append(
            f"2. Consider **{runner_up.get('model_name', 'N/A')}** as a fallback "
            f"({primary_metric}: {_safe_float(runner_up.get('metrics', {}).get(primary_metric))})."
        )
    lines.append("3. Review feature importance to identify redundant features.")
    lines.append("4. Consider ensemble methods if higher performance is required.")
    lines.append("5. Validate on production data before full deployment.")
    lines.append("")

    lines.append("---")
    lines.append(f"_Report generated by AutoML Model Selector · {ts}_")

    return "\n".join(lines)


def _primary_metric(task_type: str) -> str:
    if task_type == "classification":
        return "accuracy"
    elif task_type == "regression":
        return "rmse"
    return "score"
