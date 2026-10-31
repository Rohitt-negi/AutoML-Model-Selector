"""
AutoML Model Selector — Main Streamlit Application
A complete, interactive platform for intelligent ML model recommendation and evaluation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import io
import json
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AutoML Model Selector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "AutoML Model Selector — B.Tech Final Year Project"
    }
)

# ── Module imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from modules import data_loader, data_profiler, task_detector, recommender
from modules import preprocessing, model_trainer, evaluator
from modules import explainability, predictor, report_generator, visualizer

# ── Custom CSS ───────────────────────────────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
    /* ── Typography & Base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.875rem;
        color: #374151;
        padding: 0.35rem 0;
    }

    /* ── Main area ── */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── Headers ── */
    h1 { font-size: 1.6rem !important; font-weight: 700 !important; color: #111827 !important; }
    h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #1F2937 !important; }
    h3 { font-size: 1rem !important;   font-weight: 600 !important; color: #374151 !important; }

    /* ── Metric cards ── */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-card .metric-label {
        font-size: 0.75rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .metric-card .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-card .metric-delta {
        font-size: 0.75rem;
        color: #10B981;
        margin-top: 0.15rem;
    }

    /* ── Info/warning/error boxes ── */
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
        color: #1E40AF;
    }
    .warning-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
        color: #92400E;
    }
    .success-box {
        background: #ECFDF5;
        border-left: 4px solid #10B981;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
        color: #065F46;
    }
    .error-box {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
        color: #7F1D1D;
    }

    /* ── Recommended badge ── */
    .badge-recommended {
        display: inline-block;
        background: #D1FAE5;
        color: #065F46;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-baseline {
        display: inline-block;
        background: #E0E7FF;
        color: #3730A3;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .badge-ensemble {
        display: inline-block;
        background: #FEF3C7;
        color: #92400E;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
    }

    /* ── Workflow step indicator ── */
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        color: #9CA3AF;
    }
    .workflow-step.active {
        color: #2563EB;
        font-weight: 600;
    }
    .workflow-step.done {
        color: #10B981;
    }

    /* ── Tables ── */
    .stDataFrame {
        border: 1px solid #E5E7EB;
        border-radius: 6px;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8;
        border-color: #1D4ED8;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.9rem;
        color: #374151;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "page": "home",
        "df": None,
        "file_info": None,
        "load_error": None,
        "profile": None,
        "col_types": None,
        "target_col": None,
        "task_info": None,
        "candidates": None,
        "selected_model_ids": [],
        "eval_mode": "quick",
        "results": None,
        "best_result": None,
        "comparison_df": None,
        "preprocessor": None,
        "label_encoder": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "feature_names": None,
        "importance_df": None,
        "workflow_step": 0,   # 0=upload,1=analyze,2=target,3=recommend,4=train,5=results
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── Helpers ──────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, delta: str = ""):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def info_box(msg: str):
    st.markdown(f'<div class="info-box">ℹ️ {msg}</div>', unsafe_allow_html=True)


def warning_box(msg: str):
    st.markdown(f'<div class="warning-box">⚠️ {msg}</div>', unsafe_allow_html=True)


def success_box(msg: str):
    st.markdown(f'<div class="success-box">✅ {msg}</div>', unsafe_allow_html=True)


def error_box(msg: str):
    st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)


def step_progress():
    steps = [
        ("Upload", 0), ("Analyze", 1), ("Target", 2),
        ("Recommend", 3), ("Train", 4), ("Results", 5)
    ]
    current = st.session_state.workflow_step
    cols = st.columns(len(steps))
    for col, (label, idx) in zip(cols, steps):
        with col:
            if idx < current:
                st.markdown(f"<div style='text-align:center;font-size:0.75rem;color:#10B981;'>✓ {label}</div>", unsafe_allow_html=True)
            elif idx == current:
                st.markdown(f"<div style='text-align:center;font-size:0.75rem;color:#2563EB;font-weight:700;'>● {label}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:center;font-size:0.75rem;color:#D1D5DB;'>○ {label}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── PAGE FUNCTIONS ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── HOME ─────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown("## 🤖 AutoML Model Selector")
    st.markdown("*Intelligent model recommendation and evaluation for your datasets.*")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Supported Tasks</div>
            <div class="metric-value">7</div>
            <div class="metric-delta">Classification · Regression · Clustering · ...</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ML Algorithms</div>
            <div class="metric-value">25+</div>
            <div class="metric-delta">sklearn · XGBoost · LightGBM</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">File Formats</div>
            <div class="metric-value">5</div>
            <div class="metric-delta">CSV · Excel · JSON · TXT · ZIP</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Auto Metrics</div>
            <div class="metric-value">15+</div>
            <div class="metric-delta">Per task type, no manual config</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("&nbsp;")

    st.markdown("### How It Works")
    steps_html = """
    <div style="display:flex;align-items:flex-start;gap:1rem;flex-wrap:wrap;margin-top:0.5rem;">
    """
    workflow = [
        ("📤", "Upload Dataset", "CSV, Excel, JSON, TXT"),
        ("🔍", "Analyze Data", "Profile, quality, correlations"),
        ("🎯", "Select Target", "Auto-detect or choose manually"),
        ("💡", "Recommend Models", "Filtered by dataset characteristics"),
        ("⚙️", "Train & Evaluate", "Real metrics, no fake scores"),
        ("📊", "Compare Results", "Pick the best model"),
        ("🔮", "Make Predictions", "Use trained model interactively"),
    ]
    for icon, title, desc in workflow:
        steps_html += f"""
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:8px;padding:0.75rem;min-width:130px;flex:1;">
            <div style="font-size:1.25rem;margin-bottom:0.25rem;">{icon}</div>
            <div style="font-size:0.825rem;font-weight:600;color:#111827;">{title}</div>
            <div style="font-size:0.75rem;color:#6B7280;margin-top:0.1rem;">{desc}</div>
        </div>"""
    steps_html += "</div>"
    st.markdown(steps_html, unsafe_allow_html=True)

    st.markdown("&nbsp;")
    st.markdown("### Supported Task Types")
    tasks = {
        "🗂️ Classification": "Binary and multi-class classification with probability outputs.",
        "📈 Regression": "Continuous value prediction with residual analysis.",
        "🔵 Clustering": "Unsupervised grouping with internal validity metrics.",
        "⏱️ Time Series": "Forecasting with chronological train/test splits.",
        "📝 NLP": "Text classification and sentiment analysis.",
        "🧠 Deep Learning": "MLP for tabular, CNN/RNN for image/sequence data.",
        "🚨 Anomaly Detection": "Isolation Forest, LOF for outlier detection.",
    }
    cols = st.columns(2)
    for i, (task, desc) in enumerate(tasks.items()):
        with cols[i % 2]:
            st.markdown(f"**{task}** — {desc}")

    st.divider()
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("🚀 Get Started", type="primary", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()
    with col_b:
        info_box("Upload your first dataset to begin. Supported: CSV, Excel (.xlsx/.xls), JSON, TXT, and ZIP (image archives).")


# ── UPLOAD ───────────────────────────────────────────────────────────────────
def page_upload():
    st.markdown("## 📤 Dataset Upload")
    step_progress()
    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Upload Your Dataset")
        uploaded = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "json", "txt", "zip"],
            help="Maximum 200 MB. Supported: CSV, Excel, JSON, TXT, ZIP (image archive).",
            key="file_uploader"
        )

        if uploaded:
            with st.spinner("Loading and validating file…"):
                df, file_info, load_error = data_loader.load_dataset(uploaded)
                st.session_state.file_info = file_info
                st.session_state.load_error = load_error

            if load_error:
                error_box(load_error)
                if "unsupported" in load_error.lower() or "format" in load_error.lower():
                    info_box("**Conversion tips:** Use pandas or Excel to save your data as CSV before uploading.")
            elif df is not None:
                st.session_state.df = df
                st.session_state.workflow_step = max(st.session_state.workflow_step, 1)
                # Reset downstream state
                for k in ["profile", "target_col", "task_info", "candidates",
                          "results", "best_result", "comparison_df"]:
                    st.session_state[k] = None

                success_box(f"File loaded successfully: **{file_info.get('name')}**")

                # Quick preview
                st.markdown("**Preview (first 5 rows):**")
                st.dataframe(df.head(5), use_container_width=True)

                # Quick stats row
                n_rows, n_cols = df.shape
                missing_pct = df.isnull().mean().mean() * 100
                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_card("Rows", f"{n_rows:,}")
                with c2: metric_card("Columns", f"{n_cols}")
                with c3: metric_card("Missing", f"{missing_pct:.1f}%")
                with c4: metric_card("Size", file_info.get("size_str", "—"))

                st.markdown("&nbsp;")
                if st.button("▶ Analyze Dataset →", type="primary"):
                    st.session_state.page = "explorer"
                    st.rerun()

    with col_right:
        st.markdown("### Guidelines")
        st.markdown("""
**Accepted formats:**
- `.csv` — Comma-separated values
- `.xlsx` / `.xls` — Excel workbooks (first sheet used)
- `.json` — Array of records or column-oriented
- `.txt` — TSV or plain text
- `.zip` — Image dataset archive (organized by class folders)

**Limits:**
- Maximum file size: 200 MB
- Rows > 500,000 will trigger a performance warning
- Deeply nested JSON is not supported

**Recommendations:**
- Ensure column headers are in the first row
- Avoid merged cells in Excel files
- Use UTF-8 encoding where possible

**Unsupported formats:**
- `.parquet`, `.feather`, `.hdf5` — convert to CSV first
- `.pdf`, `.docx` — extract structured data manually
- Password-protected Excel files
        """)

        st.markdown("### Sample Datasets")
        st.markdown("Don't have a dataset? Try one of these:")
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            if st.button("🌸 Iris (Classification)", use_container_width=True):
                _load_sample("iris")
            if st.button("🏠 Boston Housing (Regression)", use_container_width=True):
                _load_sample("housing")
        with sample_col2:
            if st.button("🍷 Wine Quality (Multi-class)", use_container_width=True):
                _load_sample("wine")
            if st.button("🛒 Mall Customers (Clustering)", use_container_width=True):
                _load_sample("customers")


def _load_sample(name: str):
    """Load a built-in sample dataset."""
    from sklearn import datasets as sk_datasets
    try:
        if name == "iris":
            d = sk_datasets.load_iris(as_frame=True)
            df = d.frame
        elif name == "housing":
            # Use a safe public housing dataset
            np.random.seed(42)
            n = 506
            df = pd.DataFrame({
                "CRIM": np.random.exponential(3.5, n).round(4),
                "ZN": np.random.choice([0,12,20,25,80], n),
                "INDUS": np.random.uniform(0.5, 27, n).round(2),
                "CHAS": np.random.choice([0,1], n, p=[0.93,0.07]),
                "NOX": np.random.uniform(0.38, 0.87, n).round(3),
                "RM": np.random.normal(6.28, 0.70, n).clip(3.5, 8.8).round(3),
                "AGE": np.random.uniform(2, 100, n).round(1),
                "DIS": np.random.exponential(3.8, n).clip(1, 12).round(4),
                "RAD": np.random.choice([1,2,3,4,5,6,7,8,24], n),
                "TAX": np.random.choice([193,222,264,296,330,384,432,666,711], n),
                "PTRATIO": np.random.choice([12.6,13,13.9,14.7,15,15.3,17,18.7,19,20,21], n),
                "LSTAT": np.random.exponential(12, n).clip(1.7, 38).round(2),
                "MEDV": (np.random.normal(22.5, 9, n).clip(5, 50)).round(1)
            })
        elif name == "wine":
            d = sk_datasets.load_wine(as_frame=True)
            df = d.frame
        elif name == "customers":
            np.random.seed(42)
            n = 200
            df = pd.DataFrame({
                "CustomerID": range(1, n+1),
                "Gender": np.random.choice(["Male", "Female"], n),
                "Age": np.random.randint(18, 70, n),
                "Annual_Income_k": np.random.randint(15, 140, n),
                "Spending_Score": np.random.randint(1, 100, n),
            })
        else:
            st.error("Unknown sample dataset.")
            return

        st.session_state.df = df
        st.session_state.file_info = {
            "name": f"{name}_sample.csv",
            "extension": ".csv",
            "size_bytes": df.memory_usage(deep=True).sum(),
            "size_str": f"{df.memory_usage(deep=True).sum()/1024:.1f} KB",
        }
        st.session_state.load_error = None
        st.session_state.workflow_step = max(st.session_state.workflow_step, 1)
        for k in ["profile","target_col","task_info","candidates","results","best_result","comparison_df"]:
            st.session_state[k] = None
        success_box(f"Sample dataset '{name}' loaded.")
        st.rerun()
    except Exception as e:
        st.error(f"Could not load sample: {e}")


# ── EXPLORER ──────────────────────────────────────────────────────────────────
def page_explorer():
    if st.session_state.df is None:
        warning_box("No dataset loaded. Please upload a file first.")
        if st.button("Go to Upload"): st.session_state.page = "upload"; st.rerun()
        return

    df = st.session_state.df
    st.markdown("## 🔍 Dataset Explorer")
    step_progress()
    st.divider()

    # Profile if not done
    if st.session_state.profile is None:
        with st.spinner("Profiling dataset…"):
            profile = data_profiler.profile_dataset(df, st.session_state.target_col)
            col_types = data_profiler.get_column_types(df)
            st.session_state.profile = profile
            st.session_state.col_types = col_types
            st.session_state.workflow_step = max(st.session_state.workflow_step, 1)

    profile = st.session_state.profile
    col_types = st.session_state.col_types

    # Overview row
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card("Rows", f"{profile['n_rows']:,}")
    with c2: metric_card("Columns", f"{profile['n_cols']}")
    with c3: metric_card("Missing Values", f"{profile['total_missing_pct']:.1f}%")
    with c4: metric_card("Duplicate Rows", f"{profile['duplicate_rows']:,}")
    with c5: metric_card("Memory", profile['memory_usage'])

    st.markdown("&nbsp;")

    tabs = st.tabs(["📄 Preview", "📊 Statistics", "❓ Missing Values", "🔗 Correlations", "📦 Column Details"])

    with tabs[0]:
        st.markdown("**Dataset Preview**")
        n_preview = st.slider("Rows to preview", 5, min(100, profile['n_rows']), 10, key="preview_slider")
        st.dataframe(df.head(n_preview), use_container_width=True)

        st.markdown("**Column Types**")
        type_data = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            kind = "Numerical" if col in col_types.get('numerical', []) else \
                   "Categorical" if col in col_types.get('categorical', []) else \
                   "Datetime" if col in col_types.get('datetime', []) else \
                   "Text" if col in col_types.get('text', []) else "Other"
            miss = df[col].isnull().sum()
            miss_pct = miss / len(df) * 100
            uniq = df[col].nunique()
            type_data.append({
                "Column": col, "Type": kind, "Dtype": dtype_str,
                "Unique": uniq, "Missing": f"{miss} ({miss_pct:.1f}%)"
            })
        st.dataframe(pd.DataFrame(type_data), use_container_width=True, hide_index=True)

    with tabs[1]:
        num_cols = col_types.get('numerical', [])
        if num_cols:
            st.markdown("**Numerical Summary Statistics**")
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

            st.markdown("**Distribution Explorer**")
            selected_col = st.selectbox("Select column", num_cols, key="dist_col")
            if selected_col:
                fig = visualizer.plot_distribution(df, selected_col)
                st.plotly_chart(fig, use_container_width=True)

        cat_cols = col_types.get('categorical', [])
        if cat_cols:
            st.markdown("**Categorical Columns**")
            cat_data = []
            for c in cat_cols:
                top = df[c].value_counts().index[0] if df[c].nunique() > 0 else "—"
                cat_data.append({"Column": c, "Unique Values": df[c].nunique(), "Most Common": str(top)})
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)

            selected_cat = st.selectbox("View category distribution", cat_cols, key="cat_dist_col")
            if selected_cat and df[selected_cat].nunique() <= 50:
                fig = visualizer.plot_distribution(df, selected_cat)
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        miss_data = profile.get('missing_values', {})
        if not miss_data:
            success_box("No missing values found in this dataset.")
        else:
            miss_df = pd.DataFrame([
                {"Column": c, "Missing Count": v["count"], "Missing %": f"{v['pct']:.2f}%"}
                for c, v in miss_data.items() if v["count"] > 0
            ])
            if miss_df.empty:
                success_box("No missing values found.")
            else:
                st.dataframe(miss_df, use_container_width=True, hide_index=True)
                fig = visualizer.plot_missing_values(profile)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Warnings
                high_miss = [c for c,v in miss_data.items() if v["pct"] > 40]
                if high_miss:
                    warning_box(f"Columns with >40% missing values: {', '.join(high_miss)}. Consider dropping or imputing carefully.")

    with tabs[3]:
        num_cols = col_types.get('numerical', [])
        if len(num_cols) >= 2:
            corr_fig = visualizer.plot_correlation_heatmap(df, num_cols)
            st.plotly_chart(corr_fig, use_container_width=True)

            # Highly correlated pairs
            corr_matrix = df[num_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(num_cols)):
                for j in range(i+1, len(num_cols)):
                    if corr_matrix.iloc[i,j] > 0.85:
                        high_corr_pairs.append((num_cols[i], num_cols[j], corr_matrix.iloc[i,j]))
            if high_corr_pairs:
                warning_box(f"Highly correlated feature pairs (>0.85): " +
                           ", ".join([f"{a}↔{b} ({v:.2f})" for a,b,v in high_corr_pairs[:5]]))
        else:
            info_box("Need at least 2 numerical columns for correlation analysis.")

    with tabs[4]:
        st.markdown("**Per-Column Analysis**")
        selected_col2 = st.selectbox("Select column for details", df.columns.tolist(), key="detail_col")
        if selected_col2:
            s = df[selected_col2]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**Type:** `{s.dtype}`")
                st.markdown(f"**Non-null:** `{s.notna().sum():,}`")
                st.markdown(f"**Missing:** `{s.isnull().sum():,}`")
            with c2:
                st.markdown(f"**Unique values:** `{s.nunique():,}`")
                if s.dtype in [np.float64, np.int64, np.float32, np.int32]:
                    st.markdown(f"**Mean:** `{s.mean():.4f}`")
                    st.markdown(f"**Std:** `{s.std():.4f}`")
            with c3:
                if s.dtype in [np.float64, np.int64, np.float32, np.int32]:
                    st.markdown(f"**Min:** `{s.min()}`")
                    st.markdown(f"**Max:** `{s.max()}`")
                else:
                    st.markdown(f"**Top value:** `{s.value_counts().index[0] if s.nunique()>0 else '—'}`")

            fig = visualizer.plot_distribution(df, selected_col2)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    if st.button("▶ Select Target & Detect Task →", type="primary"):
        st.session_state.page = "task_detection"
        st.rerun()


# ── TASK DETECTION ────────────────────────────────────────────────────────────
def page_task_detection():
    if st.session_state.df is None:
        warning_box("No dataset loaded.")
        if st.button("Go to Upload"): st.session_state.page = "upload"; st.rerun()
        return

    df = st.session_state.df
    if st.session_state.profile is None:
        st.session_state.profile = data_profiler.profile_dataset(df)
        st.session_state.col_types = data_profiler.get_column_types(df)

    profile = st.session_state.profile
    col_types = st.session_state.col_types

    st.markdown("## 🎯 Problem-Type Detection")
    step_progress()
    st.divider()

    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.markdown("### Target Column")

        suggested_targets = profile.get('suggested_targets', [])
        all_cols = ["(None — Unsupervised)"] + df.columns.tolist()

        default_idx = 0
        if st.session_state.target_col and st.session_state.target_col in df.columns:
            default_idx = all_cols.index(st.session_state.target_col)
        elif suggested_targets:
            default_idx = all_cols.index(suggested_targets[0]) if suggested_targets[0] in all_cols else 0

        target_selection = st.selectbox(
            "Select target column",
            all_cols,
            index=default_idx,
            help="The column your model will predict. Choose 'None' for clustering or anomaly detection.",
            key="target_select"
        )

        target_col = None if target_selection == "(None — Unsupervised)" else target_selection

        if suggested_targets:
            info_box(f"Auto-detected candidate targets: **{', '.join(suggested_targets[:3])}**")

        st.markdown("### Task Type")
        user_hint = st.selectbox(
            "Override task type (optional)",
            ["Auto-detect", "Classification", "Regression", "Clustering",
             "Time Series", "NLP / Text Classification", "Anomaly Detection"],
            key="task_hint"
        )
        hint_map = {
            "Auto-detect": None, "Classification": "classification",
            "Regression": "regression", "Clustering": "clustering",
            "Time Series": "time_series", "NLP / Text Classification": "nlp",
            "Anomaly Detection": "anomaly_detection"
        }

        if st.button("🔍 Detect Task Type", type="primary", use_container_width=True):
            with st.spinner("Analyzing dataset and detecting task…"):
                task_info = task_detector.detect_task_type(
                    df, target_col, user_hint=hint_map[user_hint]
                )
                st.session_state.target_col = target_col
                st.session_state.task_info = task_info
                st.session_state.workflow_step = max(st.session_state.workflow_step, 2)
                # Reset downstream
                for k in ["candidates","results","best_result","comparison_df"]:
                    st.session_state[k] = None
            st.rerun()

    with col_right:
        if st.session_state.task_info:
            task_info = st.session_state.task_info
            task_type = task_info['task_type']
            desc = task_detector.get_task_description(task_type)

            task_icons = {
                "classification": "🗂️", "regression": "📈", "clustering": "🔵",
                "time_series": "⏱️", "nlp": "📝", "deep_learning": "🧠",
                "anomaly_detection": "🚨"
            }
            icon = task_icons.get(task_type, "🤖")
            conf_pct = int(task_info.get('confidence', 0.8) * 100)

            st.markdown(f"""
            <div style="background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;padding:1.25rem;margin-bottom:1rem;">
                <div style="font-size:1.5rem;margin-bottom:0.25rem;">{icon}</div>
                <div style="font-size:1rem;font-weight:700;color:#166534;">
                    Detected: {task_type.replace('_',' ').title()}
                </div>
                <div style="font-size:0.8rem;color:#15803D;margin-top:0.25rem;">
                    Confidence: {conf_pct}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            if task_info.get('sub_type'):
                st.markdown(f"**Sub-type:** {task_info['sub_type'].replace('_',' ').title()}")

            st.markdown(f"**Description:** {desc}")

            if task_info.get('reasoning'):
                st.markdown("**Reasoning:**")
                for r in task_info['reasoning']:
                    st.markdown(f"- {r}")

            if task_info.get('warnings'):
                for w in task_info['warnings']:
                    warning_box(w)

            st.divider()
            if st.button("▶ Get Model Recommendations →", type="primary"):
                st.session_state.page = "recommendations"
                st.rerun()
        else:
            info_box("Select a target column (or none for unsupervised tasks) and click 'Detect Task Type'.")

            # Show target column analysis if selected
            if target_col and target_col in df.columns:
                st.markdown("**Target Column Preview:**")
                fig = visualizer.plot_class_distribution(df[target_col], title=f"Distribution of '{target_col}'")
                st.plotly_chart(fig, use_container_width=True)


# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
def page_recommendations():
    if st.session_state.task_info is None:
        warning_box("Task type not detected yet.")
        if st.button("Go to Detection"): st.session_state.page = "task_detection"; st.rerun()
        return

    df = st.session_state.df
    profile = st.session_state.profile
    task_info = st.session_state.task_info
    task_type = task_info['task_type']

    st.markdown("## 💡 Model Recommendations")
    step_progress()
    st.divider()

    if st.session_state.candidates is None:
        with st.spinner("Generating model recommendations…"):
            candidates = recommender.get_model_candidates(
                task_type, profile, sub_type=task_info.get('sub_type')
            )
            st.session_state.candidates = candidates
            st.session_state.workflow_step = max(st.session_state.workflow_step, 3)

    candidates = st.session_state.candidates

    if not candidates:
        error_box("No suitable models found. This may indicate an unusual dataset configuration.")
        return

    # Top recommendation
    top = next((c for c in candidates if c.get('is_recommended')), candidates[0])
    st.markdown("### ⭐ Top Recommendation")
    st.markdown(f"""
    <div style="background:#FFFBEB;border:1px solid #FCD34D;border-radius:8px;padding:1.25rem;margin-bottom:1rem;">
        <div style="font-size:1rem;font-weight:700;color:#78350F;">{top['name']}</div>
        <div style="font-size:0.875rem;color:#92400E;margin-top:0.5rem;">{top.get('why_selected','')}</div>
        <div style="margin-top:0.5rem;">
            <span class="badge-{top.get('category','traditional_ml')}" style="background:#FEF3C7;color:#78350F;padding:2px 8px;border-radius:9999px;font-size:0.7rem;font-weight:600;">
                {top.get('category','').replace('_',' ').title()}
            </span>
            &nbsp;
            <span style="font-size:0.75rem;color:#6B7280;">Cost: {top.get('computational_cost','medium').title()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### All Recommended Models")
    info_box(f"Showing {len(candidates)} models suitable for **{task_type.replace('_',' ').title()}** with your dataset characteristics.")

    # Model selection for training
    st.markdown("**Select models to train:**")
    all_ids = [c['id'] for c in candidates]

    if not st.session_state.selected_model_ids:
        # Default: top 3 or all if ≤3
        st.session_state.selected_model_ids = all_ids[:min(3, len(all_ids))]

    selected_ids = []
    for c in candidates:
        badge = ""
        if c.get('is_recommended'): badge = " ⭐"
        elif c.get('category') == 'baseline': badge = " (Baseline)"

        checked = c['id'] in st.session_state.selected_model_ids
        if st.checkbox(f"{c['name']}{badge}", value=checked, key=f"chk_{c['id']}"):
            selected_ids.append(c['id'])

    st.session_state.selected_model_ids = selected_ids

    # Model cards
    st.markdown("### Model Details")
    for c in candidates:
        with st.expander(f"{'⭐ ' if c.get('is_recommended') else ''}{c['name']} — {c.get('computational_cost','').title()} Cost"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Description:** {c.get('description','')}")
                st.markdown(f"**Why selected:** {c.get('why_selected','')}")
                st.markdown("**Advantages:**")
                for adv in c.get('advantages', []):
                    st.markdown(f"  - {adv}")
            with col_b:
                st.markdown("**Limitations:**")
                for lim in c.get('limitations', []):
                    st.markdown(f"  - {lim}")
                st.markdown(f"**Minimum samples:** {c.get('min_samples', 'N/A')}")
                st.markdown(f"**Handles missing values:** {'Yes' if c.get('handles_missing') else 'No (imputation needed)'}")
                st.markdown(f"**Evaluation metrics:** {', '.join(c.get('metrics', []))}")

    st.divider()
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        eval_mode = st.selectbox(
            "Evaluation mode",
            ["quick", "standard", "advanced"],
            format_func=lambda x: {"quick": "⚡ Quick (3 models, 1 split)",
                                    "standard": "📊 Standard (all selected, 1 split)",
                                    "advanced": "🔬 Advanced (cross-validation)"}[x],
            key="eval_mode_select"
        )
        st.session_state.eval_mode = eval_mode
    with col_b:
        if st.button("⚙️ Train Selected Models →", type="primary", use_container_width=True,
                     disabled=len(st.session_state.selected_model_ids) == 0):
            st.session_state.page = "training"
            st.rerun()
    with col_c:
        if not st.session_state.selected_model_ids:
            warning_box("Select at least one model to train.")


# ── TRAINING ──────────────────────────────────────────────────────────────────
def page_training():
    if st.session_state.task_info is None or st.session_state.df is None:
        warning_box("Missing data or task info.")
        if st.button("Go Back"): st.session_state.page = "recommendations"; st.rerun()
        return

    df = st.session_state.df
    target_col = st.session_state.target_col
    task_info = st.session_state.task_info
    task_type = task_info['task_type']
    profile = st.session_state.profile
    eval_mode = st.session_state.eval_mode
    selected_ids = st.session_state.selected_model_ids

    st.markdown("## ⚙️ Model Training")
    step_progress()
    st.divider()

    if st.session_state.results is not None:
        success_box("Training already complete. View results below or proceed to comparison.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📊 View Comparison →", type="primary"):
                st.session_state.page = "comparison"
                st.rerun()
        with col_b:
            if st.button("🔄 Retrain", type="secondary"):
                st.session_state.results = None
                st.session_state.best_result = None
                st.session_state.comparison_df = None
                st.rerun()
        return

    # Show config summary
    st.markdown("### Training Configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Task", task_type.replace('_',' ').title())
    with c2: metric_card("Models", str(len(selected_ids)))
    with c3: metric_card("Mode", eval_mode.title())
    with c4: metric_card("Target", target_col or "N/A")
    st.markdown("&nbsp;")

    # Warnings
    n_rows = profile['n_rows']
    if n_rows < 100:
        warning_box(f"Dataset has only {n_rows} rows. Results may be unreliable. Consider collecting more data.")
    if n_rows > 50000 and eval_mode == "advanced":
        warning_box("Large dataset with Advanced mode may take several minutes.")

    if st.button("🚀 Start Training", type="primary", use_container_width=False):
        progress_bar = st.progress(0, text="Preparing data…")
        status_placeholder = st.empty()

        try:
            status_placeholder.info("Splitting data and building preprocessing pipeline…")
            progress_bar.progress(10, text="Splitting data…")

            eval_fn = {
                "quick": model_trainer.evaluate_quick,
                "standard": model_trainer.evaluate_standard,
                "advanced": model_trainer.evaluate_advanced,
            }[eval_mode]

            status_placeholder.info("Training models… This may take a moment.")
            progress_bar.progress(30, text="Training models…")

            results = eval_fn(
                df=df,
                target_col=target_col,
                task_type=task_type,
                profile=profile,
                model_ids=selected_ids if eval_mode != "quick" else None,
                random_state=42
            )
            progress_bar.progress(80, text="Computing metrics…")

            if not results:
                error_box("No models trained successfully. Check data and configuration.")
                return

            comparison_df = evaluator.compare_models(results, task_type)
            best_result = evaluator.get_best_model(results, task_type)

            st.session_state.results = results
            st.session_state.best_result = best_result
            st.session_state.comparison_df = comparison_df
            st.session_state.workflow_step = max(st.session_state.workflow_step, 4)

            # Compute feature importance for best model
            feature_names = best_result.get('feature_names', [])
            try:
                imp_df = explainability.get_feature_importance(best_result, feature_names)
                st.session_state.importance_df = imp_df
            except Exception:
                st.session_state.importance_df = None

            progress_bar.progress(100, text="Done!")
            status_placeholder.empty()
            success_box(f"✅ Training complete! Best model: **{best_result['model_name']}**")
            st.session_state.workflow_step = max(st.session_state.workflow_step, 5)

        except Exception as e:
            error_box(f"Training failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            return

        if st.button("📊 View Results →", type="primary"):
            st.session_state.page = "comparison"
            st.rerun()


# ── COMPARISON ────────────────────────────────────────────────────────────────
def page_comparison():
    if st.session_state.results is None:
        warning_box("No training results. Please train models first.")
        if st.button("Go to Training"): st.session_state.page = "training"; st.rerun()
        return

    results = st.session_state.results
    comparison_df = st.session_state.comparison_df
    best_result = st.session_state.best_result
    task_type = st.session_state.task_info['task_type']
    primary_metric = evaluator.get_primary_metric(task_type)

    st.markdown("## 📊 Model Comparison")
    step_progress()
    st.divider()

    # Best model highlight
    best_metrics = evaluator.format_metrics_display(best_result.get('metrics', {}), task_type)
    st.markdown("### 🏆 Best Model")
    st.markdown(f"""
    <div style="background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;padding:1.25rem;margin-bottom:1rem;">
        <div style="font-size:1.1rem;font-weight:700;color:#166534;">{best_result['model_name']}</div>
        <div style="font-size:0.85rem;color:#15803D;margin-top:0.4rem;">
            {primary_metric.upper()}: {best_metrics.get(primary_metric, '—')} &nbsp;|&nbsp;
            Training time: {best_result.get('training_time', 0):.2f}s
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Comparison table
    st.markdown("### All Models")
    if comparison_df is not None and not comparison_df.empty:
        st.dataframe(comparison_df.round(4), use_container_width=True, hide_index=True)

    # Comparison chart
    st.markdown("### Performance Comparison")
    metric_options = [m for m in comparison_df.columns if m not in ['Model', 'model_id', 'Training Time (s)']] \
                     if comparison_df is not None else [primary_metric]
    selected_metric = st.selectbox("Metric to plot", metric_options, key="comparison_metric")
    if selected_metric and comparison_df is not None:
        fig = visualizer.plot_model_comparison(comparison_df, selected_metric, task_type)
        st.plotly_chart(fig, use_container_width=True)

    # Training time
    fig_time = visualizer.plot_training_time_comparison(results)
    if fig_time:
        st.plotly_chart(fig_time, use_container_width=True)

    # Per-model evaluation details
    st.markdown("### Detailed Evaluation")
    model_names = [r['model_name'] for r in results]
    selected_model_name = st.selectbox("Select model for details", model_names, key="detail_model")
    selected_result = next((r for r in results if r['model_name'] == selected_model_name), None)

    if selected_result:
        metrics_display = evaluator.format_metrics_display(selected_result.get('metrics', {}), task_type)

        metric_items = {k: v for k, v in metrics_display.items()
                       if k not in ['confusion_matrix', 'classification_report']}
        cols = st.columns(min(len(metric_items), 4))
        for i, (k, v) in enumerate(metric_items.items()):
            with cols[i % len(cols)]:
                metric_card(k.replace('_', ' ').upper(), str(v))

        if task_type == 'classification':
            y_true = selected_result.get('y_test')
            y_pred = selected_result.get('y_pred')
            if y_true is not None and y_pred is not None:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Confusion Matrix**")
                    fig_cm = visualizer.plot_confusion_matrix(y_true, y_pred)
                    st.plotly_chart(fig_cm, use_container_width=True)
                with col_b:
                    y_proba = selected_result.get('y_pred_proba')
                    if y_proba is not None:
                        import numpy as _np
                        n_classes = len(_np.unique(y_true))
                        if n_classes == 2:
                            st.markdown("**ROC Curve**")
                            fig_roc = visualizer.plot_roc_curve(y_true, y_proba, n_classes)
                            if fig_roc:
                                st.plotly_chart(fig_roc, use_container_width=True)

            cr = selected_result.get('metrics', {}).get('classification_report')
            if cr:
                with st.expander("Classification Report"):
                    st.text(cr)

        elif task_type == 'regression':
            y_true = selected_result.get('y_test')
            y_pred = selected_result.get('y_pred')
            if y_true is not None and y_pred is not None:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Residual Plot**")
                    fig_res = visualizer.plot_residuals(y_true, y_pred)
                    st.plotly_chart(fig_res, use_container_width=True)
                with col_b:
                    st.markdown("**Actual vs Predicted**")
                    fig_ap = visualizer.plot_actual_vs_predicted(y_true, y_pred)
                    st.plotly_chart(fig_ap, use_container_width=True)

        if selected_result.get('cv_scores'):
            cv = selected_result['cv_scores']
            info_box(f"Cross-validation ({len(cv)} folds) — Mean: {np.mean(cv):.4f} ± {np.std(cv):.4f}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔮 Prediction Playground →", type="primary"):
            st.session_state.page = "prediction"
            st.rerun()
    with col_b:
        if st.button("🧠 Explainability →"):
            st.session_state.page = "explainability"
            st.rerun()


# ── PREDICTION PLAYGROUND ─────────────────────────────────────────────────────
def page_prediction():
    if st.session_state.best_result is None:
        warning_box("No trained model available. Please train models first.")
        if st.button("Go to Training"): st.session_state.page = "training"; st.rerun()
        return

    df = st.session_state.df
    results = st.session_state.results
    task_type = st.session_state.task_info['task_type']
    target_col = st.session_state.target_col

    st.markdown("## 🔮 Prediction Playground")
    step_progress()
    st.divider()

    model_names = [r['model_name'] for r in results]
    selected_name = st.selectbox("Select model for predictions", model_names, key="pred_model")
    selected_result = next((r for r in results if r['model_name'] == selected_name), None)

    if not selected_result:
        return

    pipeline = selected_result.get('pipeline')
    feature_names = selected_result.get('feature_names', [])
    label_encoder = selected_result.get('label_encoder')

    if not pipeline or not feature_names:
        error_box("Pipeline not available for this model.")
        return

    feature_cols = [c for c in feature_names if c != target_col]

    st.markdown("### Enter Feature Values")
    info_box("Enter values for each feature. Leave blank to use the median/mode from the training data.")

    input_data = {}
    col_types = st.session_state.col_types or {}
    num_cols = col_types.get('numerical', [])
    cat_cols = col_types.get('categorical', [])

    # Determine reasonable input columns
    input_cols = [c for c in df.columns if c != target_col and c in feature_names]

    n_cols = min(3, len(input_cols))
    if n_cols == 0:
        error_box("No input features available.")
        return

    grid_cols = st.columns(n_cols)
    for i, col_name in enumerate(input_cols):
        with grid_cols[i % n_cols]:
            if col_name in num_cols:
                col_min = float(df[col_name].min()) if df[col_name].notna().any() else 0.0
                col_max = float(df[col_name].max()) if df[col_name].notna().any() else 1.0
                col_mean = float(df[col_name].mean()) if df[col_name].notna().any() else 0.0
                val = st.number_input(
                    col_name, value=col_mean,
                    min_value=col_min - abs(col_min),
                    max_value=col_max + abs(col_max),
                    key=f"pred_{col_name}"
                )
                input_data[col_name] = val
            elif col_name in cat_cols:
                options = df[col_name].dropna().unique().tolist()
                val = st.selectbox(col_name, options, key=f"pred_{col_name}")
                input_data[col_name] = val
            else:
                val = st.text_input(col_name, key=f"pred_{col_name}")
                input_data[col_name] = val

    st.markdown("&nbsp;")
    if st.button("🔮 Predict", type="primary"):
        try:
            is_valid, errors = predictor.validate_input(input_data, feature_names, df, task_type)
            if not is_valid:
                for err in errors:
                    warning_box(err)

            result = predictor.predict_single(
                pipeline, input_data, task_type, feature_names, label_encoder
            )
            output = predictor.format_prediction_output(result, task_type, label_encoder)

            st.markdown("### Prediction Result")
            pred_val = output.get('prediction', '—')
            st.markdown(f"""
            <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:1.5rem;text-align:center;">
                <div style="font-size:0.875rem;color:#1E40AF;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Prediction</div>
                <div style="font-size:2rem;font-weight:700;color:#1E3A8A;margin-top:0.5rem;">{pred_val}</div>
            </div>
            """, unsafe_allow_html=True)

            if output.get('probabilities'):
                st.markdown("**Class Probabilities:**")
                prob_df = pd.DataFrame([output['probabilities']]).T.reset_index()
                prob_df.columns = ['Class', 'Probability']
                prob_df['Probability'] = prob_df['Probability'].round(4)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

        except Exception as e:
            error_box(f"Prediction failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


# ── EXPLAINABILITY ────────────────────────────────────────────────────────────
def page_explainability():
    if st.session_state.best_result is None:
        warning_box("No trained model available.")
        if st.button("Go to Training"): st.session_state.page = "training"; st.rerun()
        return

    results = st.session_state.results
    task_type = st.session_state.task_info['task_type']
    df = st.session_state.df

    st.markdown("## 🧠 Model Explainability")
    step_progress()
    st.divider()

    model_names = [r['model_name'] for r in results]
    selected_name = st.selectbox("Select model", model_names, key="expl_model")
    selected_result = next((r for r in results if r['model_name'] == selected_name), None)

    if not selected_result:
        return

    pipeline = selected_result.get('pipeline')
    feature_names = selected_result.get('feature_names', [])

    tabs = st.tabs(["📊 Feature Importance", "🔀 Permutation Importance", "📝 Coefficients"])

    with tabs[0]:
        try:
            imp_df = explainability.get_feature_importance(selected_result, feature_names)
            if imp_df is not None and not imp_df.empty:
                st.markdown("**Feature Importance (Model-Based)**")
                explanation = explainability.explain_feature_importance(imp_df, task_type)
                info_box(explanation)
                fig = visualizer.plot_feature_importance(imp_df)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(imp_df.head(20).round(4), use_container_width=True, hide_index=True)
            else:
                info_box("Feature importance not available for this model type.")
        except Exception as e:
            warning_box(f"Could not compute feature importance: {str(e)}")

    with tabs[1]:
        X_test = selected_result.get('X_test')
        y_test = selected_result.get('y_test')
        if X_test is not None and y_test is not None:
            if st.button("Compute Permutation Importance", key="perm_imp_btn"):
                with st.spinner("Computing permutation importance (may take a moment)…"):
                    try:
                        perm_df = explainability.compute_permutation_importance(
                            pipeline, X_test, y_test, feature_names, task_type
                        )
                        if perm_df is not None and not perm_df.empty:
                            fig = visualizer.plot_feature_importance(perm_df)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            warning_box("Permutation importance returned no results.")
                    except Exception as e:
                        warning_box(f"Permutation importance failed: {str(e)}")
        else:
            info_box("Test data not available for permutation importance.")

    with tabs[2]:
        try:
            coef_df = explainability.get_model_coefficients(pipeline, feature_names)
            if coef_df is not None and not coef_df.empty:
                st.markdown("**Model Coefficients**")
                info_box("Positive coefficients increase the prediction; negative ones decrease it.")
                fig = visualizer.plot_feature_importance(coef_df.rename(columns={'coefficient': 'importance'}))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(coef_df.round(4), use_container_width=True, hide_index=True)
            else:
                info_box("Coefficient interpretation is not available for this model type. It works with linear models (Logistic Regression, Ridge, Lasso, etc.).")
        except Exception as e:
            warning_box(f"Could not extract coefficients: {str(e)}")


# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
def page_download():
    st.markdown("## 📥 Download Results")
    step_progress()
    st.divider()

    profile = st.session_state.profile
    results = st.session_state.results
    comparison_df = st.session_state.comparison_df
    best_result = st.session_state.best_result
    task_type = st.session_state.task_info['task_type'] if st.session_state.task_info else None

    if not any([profile, results]):
        warning_box("Nothing to download yet. Complete the analysis and training first.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Dataset & Analysis")

        if profile:
            report_md = report_generator.generate_dataset_report(profile, st.session_state.df)
            st.download_button(
                "📄 Dataset Analysis Report (Markdown)",
                data=report_md.encode(),
                file_name="dataset_analysis_report.md",
                mime="text/markdown",
                use_container_width=True
            )

        if comparison_df is not None:
            csv_bytes = report_generator.generate_model_comparison_csv(comparison_df)
            st.download_button(
                "📊 Model Comparison (CSV)",
                data=csv_bytes,
                file_name="model_comparison.csv",
                mime="text/csv",
                use_container_width=True
            )

        if results and task_type:
            eval_report = report_generator.generate_evaluation_report(results, task_type, profile)
            st.download_button(
                "📋 Evaluation Report (Markdown)",
                data=eval_report.encode(),
                file_name="evaluation_report.md",
                mime="text/markdown",
                use_container_width=True
            )

    with col_right:
        st.markdown("### Models & Artifacts")

        if best_result and best_result.get('pipeline'):
            try:
                model_bytes = report_generator.export_model_artifact(
                    best_result['pipeline'],
                    best_result.get('model_id', 'model'),
                    {'task_type': task_type, 'model_name': best_result['model_name']}
                )
                st.download_button(
                    f"🤖 Best Model Pipeline ({best_result['model_name']})",
                    data=model_bytes,
                    file_name=f"{best_result.get('model_id','model')}_pipeline.joblib",
                    mime="application/octet-stream",
                    use_container_width=True
                )
                info_box("Download the pipeline and load it with `joblib.load('pipeline.joblib')` for future predictions.")
            except Exception as e:
                warning_box(f"Could not export model: {str(e)}")

        if profile and results and task_type:
            try:
                pdf_bytes = report_generator.generate_summary_pdf(
                    profile, results, best_result, task_type, output_path=None
                )
                if pdf_bytes:
                    st.download_button(
                        "📑 Summary Report (PDF)",
                        data=pdf_bytes,
                        file_name="automl_summary_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                warning_box(f"PDF generation failed: {str(e)}. Try the Markdown reports instead.")


# ── ABOUT ─────────────────────────────────────────────────────────────────────
def page_about():
    st.markdown("## ℹ️ About AutoML Model Selector")
    st.divider()
    st.markdown("""
### Project Overview

**AutoML Model Selector** is a final-year B.Tech Data Science project that helps students, beginners, and analysts 
choose the right machine learning model for their dataset — without needing to manually test every algorithm.

### Architecture

```
automl-model-selector/
├── app.py                    ← Main Streamlit application
├── modules/
│   ├── data_loader.py        ← Safe file loading (CSV, Excel, JSON, TXT, ZIP)
│   ├── data_profiler.py      ← Dataset analysis and profiling
│   ├── task_detector.py      ← Problem type detection
│   ├── recommender.py        ← Model candidate selection
│   ├── preprocessing.py      ← Feature engineering pipelines
│   ├── model_trainer.py      ← Training and validation orchestration
│   ├── evaluator.py          ← Metrics and model comparison
│   ├── explainability.py     ← Feature importance and SHAP
│   ├── predictor.py          ← Interactive predictions
│   ├── report_generator.py   ← PDF and CSV exports
│   └── visualizer.py         ← All Plotly visualizations
├── tests/                    ← pytest test suite
├── sample_data/              ← Sample datasets for testing
├── requirements.txt          ← Python dependencies
└── README.md                 ← Installation and usage guide
```

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | Streamlit, Custom CSS |
| Data | Pandas, NumPy, SciPy |
| ML | scikit-learn, XGBoost, LightGBM |
| Visualization | Plotly, Matplotlib, Seaborn |
| NLP | NLTK, TF-IDF |
| Export | joblib, fpdf2, ReportLab |
| Testing | pytest |

### Supported Tasks

- **Classification:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting, XGBoost
- **Regression:** Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, SVR
- **Clustering:** K-Means, DBSCAN, Agglomerative, Gaussian Mixture
- **Time Series:** Baseline, ARIMA, ML-based forecasting
- **NLP:** TF-IDF + Logistic Regression, Naive Bayes text classification
- **Anomaly Detection:** Isolation Forest, Local Outlier Factor

### Limitations

- Deep learning (CNN, LSTM) requires TensorFlow/PyTorch installation
- Very large datasets (>500k rows) may be slow in Advanced evaluation mode
- Image datasets require organized ZIP archives (class folders)
- Transformer models (BERT, etc.) require additional setup
- PDF export requires fpdf2

### Future Scope

- React + FastAPI backend architecture
- AutoML hyperparameter search (Optuna/Ray Tune)
- MLflow experiment tracking
- REST API for model serving
- Docker deployment support
- Online learning and incremental updates

---

**Developed as a B.Tech Final Year Project | Data Science**  
*Uses real model training and real evaluation metrics. No fake scores.*
    """)


# ══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR & ROUTING ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("### 🤖 AutoML Model Selector")
        st.markdown("---")

        # Status indicators
        df_loaded = st.session_state.df is not None
        profile_done = st.session_state.profile is not None
        task_done = st.session_state.task_info is not None
        trained = st.session_state.results is not None

        def nav_item(icon, label, page_key, enabled=True):
            if enabled:
                if st.button(f"{icon} {label}", use_container_width=True,
                             type="primary" if st.session_state.page == page_key else "secondary"):
                    st.session_state.page = page_key
                    st.rerun()
            else:
                st.button(f"{icon} {label}", use_container_width=True, disabled=True)

        nav_item("🏠", "Home", "home")
        nav_item("📤", "Upload Dataset", "upload")
        nav_item("🔍", "Dataset Explorer", "explorer", df_loaded)
        nav_item("🎯", "Task Detection", "task_detection", profile_done)
        nav_item("💡", "Model Recommendations", "recommendations", task_done)
        nav_item("⚙️", "Train Models", "training", task_done)
        nav_item("📊", "Model Comparison", "comparison", trained)
        nav_item("🔮", "Prediction Playground", "prediction", trained)
        nav_item("🧠", "Explainability", "explainability", trained)
        nav_item("📥", "Download Results", "download", profile_done)
        nav_item("ℹ️", "About", "about")

        st.markdown("---")

        # Current dataset info
        if st.session_state.df is not None:
            fi = st.session_state.file_info or {}
            st.markdown("**Loaded Dataset**")
            st.markdown(f"📁 `{fi.get('name', 'dataset')}`")
            df = st.session_state.df
            st.markdown(f"🔢 `{df.shape[0]:,}` rows × `{df.shape[1]}` cols")
            if st.session_state.target_col:
                st.markdown(f"🎯 Target: `{st.session_state.target_col}`")
            if st.session_state.task_info:
                tt = st.session_state.task_info['task_type'].replace('_',' ').title()
                st.markdown(f"📌 Task: `{tt}`")

        st.markdown("---")
        st.markdown("<div style='font-size:0.7rem;color:#9CA3AF;'>AutoML Model Selector v1.0</div>", unsafe_allow_html=True)


def main():
    load_css()
    init_session_state()
    sidebar()

    page = st.session_state.page

    page_map = {
        "home": page_home,
        "upload": page_upload,
        "explorer": page_explorer,
        "task_detection": page_task_detection,
        "recommendations": page_recommendations,
        "training": page_training,
        "comparison": page_comparison,
        "prediction": page_prediction,
        "explainability": page_explainability,
        "download": page_download,
        "about": page_about,
    }

    fn = page_map.get(page, page_home)
    fn()


if __name__ == "__main__":
    main()

