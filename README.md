# AutoML Model Selector — Intelligent Model Recommendation & Evaluation Platform

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20Scikit--Learn-orange)](https://streamlit.io/)
[![Project Grade](https://img.shields.io/badge/Type-B.Tech%20Final%20Year%20Data%20Science%20Project-green)](#)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📌 Executive Summary

**AutoML Model Selector** is an automated machine learning recommendation and evaluation platform developed as a final-year B.Tech Capstone Project in Data Science. It bridges the gap between raw data collection and algorithm deployment by eliminating manual trial-and-error testing while strictly preserving sound data science principles (no data leakage, stratified validation, and interpretable metric attribution).

Unlike superficial AutoML wrappers or template dashboards, this platform performs **statistically sound dataset profiling**, **heuristic problem-type inference**, **context-aware algorithm shortlisting**, **isolated train-test preprocessing pipelines**, **multi-metric performance benchmarking**, **interactive single/batch predictions**, and **model explainability via feature attributions and SHAP**.

---

## 🏗️ System Architecture

The project is structured according to modular engineering principles, separating file ingest, profiling, recommendation, execution, evaluation, explainability, and presentation layers:

```
automl-model-selector/
│
├── app.py                         # Main interactive Streamlit application & routing
├── requirements.txt               # Pinned dependencies for reproducible environments
├── run.sh                         # Turnkey automated setup and launch script
│
├── modules/                       # Decoupled backend business logic
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # Multi-format safe parser (CSV, Excel, JSON, TXT, ZIP)
│   ├── data_profiler.py           # Statistical dataset profiling, outlier detection, IQR
│   ├── task_detector.py           # Problem-type & target inference engine
│   ├── recommender.py             # Knowledge-based algorithm candidate selector
│   ├── preprocessing.py           # Leak-free ColumnTransformer & encoding pipelines
│   ├── model_trainer.py           # Cross-validated training orchestrator
│   ├── evaluator.py               # Multi-metric computation & ranking engine
│   ├── explainability.py          # Feature importance, coefficients & SHAP explainer
│   ├── predictor.py               # Live inference engine with schema validation
│   ├── report_generator.py        # Automated Markdown, CSV & PDF export generation
│   └── visualizer.py              # Consistent Plotly visualization components
│
├── sample_data/                   # Curated benchmark datasets for instant evaluation
│   ├── customer_churn.csv         # Classification (binary, imbalanced, mixed features)
│   ├── car_prices.csv             # Regression (continuous pricing, multi-feature)
│   ├── product_reviews.csv        # Natural Language Processing (sentiment classification)
│   └── monthly_sales.csv          # Time-Series Forecasting (temporal sequences)
│
└── tests/                         # Pytest automated testing suite
    ├── __init__.py
    ├── test_data_loader.py        # Ingestion & edge case unit tests
    ├── test_data_profiler.py      # Profiling & correlation tests
    ├── test_task_detector.py      # Target and task inference tests
    ├── test_recommender.py        # Recommendation filtering tests
    ├── test_preprocessing.py      # Pipeline & leakage prevention tests
    ├── test_model_trainer.py      # Training execution tests
    ├── test_evaluator.py          # Scoring and ranking tests
    ├── test_predictor.py          # Prediction playground tests
    ├── test_explainability.py     # Attribution & coefficient tests
    └── test_report_generator.py   # Document synthesis tests
```

---

## 🔄 End-to-End Workflow Pipeline

```mermaid
flowchart LR
    A[1. Upload Dataset] --> B[2. Data Profiler]
    B --> C[3. Task Detector]
    C --> D[4. Recommendation Engine]
    D --> E[5. Pipeline & Trainer]
    E --> F[6. Fair Evaluator]
    F --> G[7. Explainability & Prediction]
```

1. **Upload Dataset:** Supports CSV, Excel (.xlsx/.xls), JSON, TXT, and ZIP (image archive inspection). Validates size (max 200MB), character encodings via `chardet`, and file corruption.
2. **Analyze & Profile:** Computes cardinality, missing rates, duplicate counts, IQR outlier fences, Pearson correlation matrices, and distribution metrics.
3. **Task & Target Detection:** Determines problem types (**Classification**, **Regression**, **Clustering**, **Time Series**, **NLP**, or **Anomaly Detection**) with confidence scores and reasoning explanations.
4. **Model Recommendation:** Analyzes sample size, feature counts, missing ratios, and class balance to recommend suitable algorithms with pros, cons, and computational costs.
5. **Model Training:** Executes in three user-selectable modes:
   - **Quick Evaluation:** Trains 3 representative baseline models on a single validation split.
   - **Standard Evaluation:** Trains the entire recommended candidate pool.
   - **Advanced Evaluation:** Performs 5-Fold Stratified / K-Fold Cross Validation.
6. **Model Evaluation & Comparison:** Ranks models fairly using matching test splits and task-specific primary metrics.
7. **Prediction & Explainability:** Interactive single-instance and batch predictions, accompanied by global feature importance and linear coefficients.

---

## 🤖 Supported Algorithms & Tasks

| Problem Type | Baseline Models | Traditional ML | Ensemble / Advanced |
| :--- | :--- | :--- | :--- |
| **Classification** | Logistic Regression, Naive Bayes | Decision Tree, SVM, KNN | Random Forest, Gradient Boosting, XGBoost |
| **Regression** | Linear Regression (OLS) | Ridge, Lasso, Decision Tree, SVR | Random Forest Regressor, Gradient Boosting |
| **Clustering** | K-Means | Agglomerative Clustering | DBSCAN, Gaussian Mixture Models (GMM) |
| **Time Series** | Naive Persistence / Rolling Mean | ARIMA / SARIMA | Lagged Random Forest Regressor |
| **NLP** | TF-IDF + Naive Bayes | TF-IDF + Logistic Regression | TF-IDF + Random Forest Classifier |
| **Anomaly Detection** | Simple Outlier Threshold | Local Outlier Factor (LOF) | Isolation Forest |

---

## 🛡️ Rigorous Data Science Standards

- **Zero Data Leakage:** Preprocessors (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`) are fitted strictly on `X_train` and applied down to `X_test` and inference payloads.
- **Stratified Partitioning:** Supervised classification preserves class proportions using `StratifiedShuffleSplit` / stratified `train_test_split`.
- **Target Encoding Protection:** Classification targets are mapped with fitted `LabelEncoder` objects and decoupled from feature transformation spaces.
- **Genuine Execution:** No dummy numbers, mocks, or synthetic delays. Real models are trained and scored against ground-truth validation splits.

---

## 🚀 Installation & Local Execution

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12
- `pip` and `virtualenv`

### Quick Start (One Command)

```bash
cd automl-model-selector
chmod +x run.sh
./run.sh
```

### Manual Setup

1. **Clone or Navigate to the Directory:**
   ```bash
   cd /path/to/automl-model-selector
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # On Linux/macOS
   # .\venv\Scripts\activate      # On Windows PowerShell
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```
   The dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 🧪 Running Automated Unit Tests

The test suite covers data loading, task detection, preprocessing, model training, evaluation metrics, and prediction.

Run the test suite using `pytest`:

```bash
pytest tests/ -v
```

---

## 📊 Sample Datasets Provided

For instant testing and evaluation without needing an external dataset:
- `sample_data/customer_churn.csv` (Binary Classification, 30 rows, mixed categorical & numerical features)
- `sample_data/car_prices.csv` (Regression, 25 rows, continuous vehicle valuation)
- `sample_data/product_reviews.csv` (NLP Text Classification, 20 sentiment reviews)
- `sample_data/monthly_sales.csv` (Time Series, 30 sequential monthly revenue observations)

Or click on any of the built-in sample buttons directly on the **Dataset Upload** page.

---

## 📄 Export & Deliverables

Users can download:
1. **Dataset Analysis Report:** Full Markdown audit covering column classifications, null rates, and distribution summaries.
2. **Model Comparison Matrix:** Filterable `.csv` sheet with accuracy, F1, precision, recall, MAE, RMSE, and training duration.
3. **Trained Pipeline Artifact:** Serialized `.joblib` model containing the full transformer pipeline ready for deployment via `joblib.load()`.
4. **Comprehensive PDF Summary:** Formatted academic report generated with `fpdf2`.

---

## 🔮 Future Scope & Extension
- Integration with Optuna for Bayesian hyperparameter search.
- Decoupling into a standalone FastAPI backend with a React / Next.js frontend.
- Cloud deployment recipes for Docker, AWS ECS, and Hugging Face Spaces.
- Native fine-tuning integration with Hugging Face Transformers (`DistilBERT`).

---

**Developed for B.Tech Final Year Capstone Project in Data Science & Machine Learning**  
*Academic Year 2025–2026*

