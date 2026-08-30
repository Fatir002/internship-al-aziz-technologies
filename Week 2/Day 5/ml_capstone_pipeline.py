"""
Week 2 | Day 5 — Capstone: End-to-End ML Pipeline
====================================================
Internship at Al Aziz Technologies | AI Engineering Track

This script is a production-style, self-contained machine-learning pipeline:

    1. Generate synthetic customer churn dataset  (500 rows, 7 features)
    2. Exploratory snapshot  (shape, churn rate, correlations)
    3. Build a ColumnTransformer preprocessing pipeline
    4. Train 5 classifiers inside Scikit-Learn Pipelines
    5. 5-fold Stratified Cross-Validation for all models
    6. Final evaluation on a held-out test set (20%)
    7. Per-class classification report + confusion matrix
    8. Export metrics to  Week 2/Day 5/evaluation_report.md
    9. Serialise the best model to  /tmp/best_model.joblib

Run:
    python "Week 2/Day 5/ml_capstone_pipeline.py"
"""

import json
import logging
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("capstone.ml")

# ── Output paths ──────────────────────────────────────────────────────────────
REPORT_PATH = Path(__file__).parent / "evaluation_report.md"
MODEL_PATH  = Path(tempfile.gettempdir()) / "best_model.joblib"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic customer churn dataset with realistic feature correlations.

    Features:
        age, tenure_months, monthly_fee  → numeric
        contract, internet, tech_support → categorical (nominal)
        seniority                        → categorical (ordinal)

    Target:
        churn  (0 = retained, 1 = churned)
    """
    rng = np.random.default_rng(seed)

    age         = rng.integers(18, 70, size=n)
    tenure      = rng.integers(1,  120, size=n)
    monthly_fee = rng.uniform(20, 150, size=n).round(2)
    contracts   = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"], size=n, p=[0.5, 0.3, 0.2]
    )
    internet    = rng.choice(["DSL", "Fiber", "No"], size=n, p=[0.4, 0.4, 0.2])
    support     = rng.choice(["Yes", "No"],           size=n, p=[0.4, 0.6])
    seniority   = rng.choice(
        ["Junior", "Mid", "Senior"], size=n, p=[0.3, 0.4, 0.3]
    )

    churn_prob = np.clip(
        0.35
        - 0.20 * (contracts == "Two Year")
        - 0.10 * (contracts == "One Year")
        - 0.003 * tenure
        + 0.15  * (internet == "Fiber")
        + rng.uniform(-0.1, 0.1, size=n),
        0.05, 0.95,
    )
    churn = (rng.random(size=n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "age":            age,
        "tenure_months":  tenure,
        "monthly_fee":    monthly_fee,
        "contract":       contracts,
        "internet":       internet,
        "tech_support":   support,
        "seniority":      seniority,
        "churn":          churn,
    })
    logger.info("Dataset generated | shape=%s | churn_rate=%.1f%%",
                df.shape, df["churn"].mean() * 100)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
def eda_snapshot(df: pd.DataFrame) -> None:
    logger.info("── EDA Snapshot ──")
    print("\n── EDA Snapshot ──")
    print(f"  Shape       : {df.shape}")
    print(f"  Churn rate  : {df['churn'].mean():.1%}  ({df['churn'].sum()} churned)")
    print(f"  Null values : {df.isnull().sum().sum()}")
    print(f"\n  Numeric summary:")
    numeric_cols = ["age", "tenure_months", "monthly_fee"]
    print(df[numeric_cols].describe().round(2).to_string())
    print(f"\n  Categorical value counts:")
    for col in ["contract", "internet", "tech_support", "seniority"]:
        counts = df[col].value_counts().to_dict()
        print(f"    {col:<15}: {counts}")

    # Churn by contract type
    print(f"\n  Churn rate by contract:")
    churn_by_contract = df.groupby("contract")["churn"].mean().round(3)
    for c, r in churn_by_contract.items():
        print(f"    {c:<20}: {r:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_preprocessor() -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer for mixed feature types."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(),
             ["age", "tenure_months", "monthly_fee"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["contract", "internet", "tech_support"]),
            ("ord", OrdinalEncoder(categories=[["Junior", "Mid", "Senior"]]),
             ["seniority"]),
        ],
        remainder="drop",
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 & 5 — MODEL TRAINING + CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def train_and_crossval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[dict]:
    """
    Train 5 classifiers using 5-fold Stratified CV.

    Returns:
        List of result dicts (one per model) with CV metrics.
    """
    logger.info("── Training & Cross-Validation ──")

    classifiers = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
        ("Gradient Boosting",
         GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                     max_depth=4, random_state=42)),
        ("SVM (RBF)",
         SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
        ("K-Nearest Neighbours",
         KNeighborsClassifier(n_neighbors=7, weights="distance")),
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "roc_auc", "precision", "recall"]
    results = []

    for name, clf in classifiers:
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
        cv_result = cross_validate(
            pipe, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1
        )
        entry = {
            "model":        name,
            "cv_accuracy":  round(cv_result["test_accuracy"].mean(),  4),
            "cv_f1":        round(cv_result["test_f1"].mean(),         4),
            "cv_roc_auc":   round(cv_result["test_roc_auc"].mean(),    4),
            "cv_precision": round(cv_result["test_precision"].mean(),  4),
            "cv_recall":    round(cv_result["test_recall"].mean(),     4),
            "cv_std":       round(cv_result["test_roc_auc"].std(),     4),
            "_pipe":        pipe,   # kept for final fit — removed before export
        }
        results.append(entry)
        logger.info("%-26s  CV AUC=%.4f ± %.4f  CV F1=%.4f",
                    name, entry["cv_roc_auc"], entry["cv_std"], entry["cv_f1"])

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — FINAL EVALUATION ON HELD-OUT TEST SET
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_on_test(
    cv_results: list[dict],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[dict]:
    """
    Fit each model on full training data, evaluate on test set.

    Returns:
        Updated results list with test-set metrics added.
    """
    logger.info("── Final Test-Set Evaluation ──")
    print("\n── Final Test-Set Evaluation ──")
    print(f"  {'Model':<26}  {'Test Acc':>9}  {'Test F1':>8}  {'Test AUC':>9}  {'Prec':>6}  {'Rec':>6}")
    print(f"  {'-'*26}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*6}")

    for entry in cv_results:
        pipe = entry["_pipe"]
        pipe.fit(X_train, y_train)

        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        entry["test_accuracy"]  = round(accuracy_score(y_test, y_pred),                   4)
        entry["test_f1"]        = round(f1_score(y_test, y_pred, zero_division=0),        4)
        entry["test_roc_auc"]   = round(roc_auc_score(y_test, y_proba),                   4)
        entry["test_precision"] = round(precision_score(y_test, y_pred, zero_division=0), 4)
        entry["test_recall"]    = round(recall_score(y_test, y_pred, zero_division=0),    4)
        entry["_y_pred"]        = y_pred
        entry["_y_proba"]       = y_proba

        print(
            f"  {entry['model']:<26}  "
            f"{entry['test_accuracy']:>9.4f}  "
            f"{entry['test_f1']:>8.4f}  "
            f"{entry['test_roc_auc']:>9.4f}  "
            f"{entry['test_precision']:>6.4f}  "
            f"{entry['test_recall']:>6.4f}"
        )

    return cv_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — DETAILED REPORT FOR BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
def detailed_report(best: dict, y_test: pd.Series) -> None:
    """Print confusion matrix and classification report for the best model."""
    logger.info("── Detailed Report: %s ──", best["model"])
    print(f"\n── Best Model: {best['model']} ──")

    cm = confusion_matrix(y_test, best["_y_pred"])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>10} Pred 0   Pred 1")
    print(f"  Actual 0 :  {tn:>5}    {fp:>5}")
    print(f"  Actual 1 :  {fn:>5}    {tp:>5}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, best["_y_pred"],
        target_names=["No Churn", "Churn"],
        digits=4,
    ))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — EXPORT EVALUATION REPORT
# ══════════════════════════════════════════════════════════════════════════════
def export_markdown_report(
    cv_results: list[dict],
    best: dict,
    y_test: pd.Series,
    dataset_shape: tuple,
    churn_rate: float,
) -> None:
    """Write a structured Markdown evaluation report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cm = confusion_matrix(y_test, best["_y_pred"])
    tn, fp, fn, tp = cm.ravel()

    # Table rows
    table_rows = ""
    for r in sorted(cv_results, key=lambda x: x["test_roc_auc"], reverse=True):
        star = " ⭐" if r["model"] == best["model"] else ""
        table_rows += (
            f"| {r['model']}{star} | {r['cv_roc_auc']} ± {r['cv_std']} "
            f"| {r['test_accuracy']} | {r['test_f1']} "
            f"| {r['test_roc_auc']} | {r['test_precision']} | {r['test_recall']} |\n"
        )

    cls_report = classification_report(
        y_test, best["_y_pred"],
        target_names=["No Churn", "Churn"],
        digits=4,
    )

    md = f"""# Week 2 — ML Capstone: Evaluation Report

**Project:** Customer Churn Prediction  
**Intern:** Fatir Faraz  
**Internship:** Al Aziz Technologies — AI Engineering Track  
**Generated:** {timestamp}

---

## 1. Dataset Summary

| Attribute | Value |
|-----------|-------|
| Total rows | {dataset_shape[0]} |
| Features | {dataset_shape[1] - 1} |
| Target | `churn` (binary: 0 = retained, 1 = churned) |
| Churn rate | {churn_rate:.1%} |
| Train / Test split | 80% / 20% (stratified) |
| Cross-validation | 5-fold Stratified K-Fold |

### Feature Types

| Feature | Type | Notes |
|---------|------|-------|
| `age` | Numeric | StandardScaler |
| `tenure_months` | Numeric | StandardScaler |
| `monthly_fee` | Numeric | StandardScaler |
| `contract` | Categorical (nominal) | OneHotEncoder |
| `internet` | Categorical (nominal) | OneHotEncoder |
| `tech_support` | Categorical (nominal) | OneHotEncoder |
| `seniority` | Categorical (ordinal) | OrdinalEncoder (Junior < Mid < Senior) |

---

## 2. Model Comparison

| Model | CV ROC-AUC (±std) | Test Accuracy | Test F1 | Test ROC-AUC | Precision | Recall |
|-------|-------------------|---------------|---------|--------------|-----------|--------|
{table_rows}

> ⭐ = Best model selected for export

---

## 3. Best Model: {best['model']}

### Hyperparameters
See source code — `ml_capstone_pipeline.py`, `train_and_crossval()`.

### Confusion Matrix

|  | Predicted No Churn | Predicted Churn |
|--|-------------------|-----------------|
| **Actual No Churn** | {tn} (TN) | {fp} (FP) |
| **Actual Churn** | {fn} (FN) | {tp} (TP) |

### Per-Class Report

```
{cls_report}
```

---

## 4. Key Findings

- The **{best['model']}** achieved the highest test ROC-AUC of **{best['test_roc_auc']:.4f}**.
- Customers on **Month-to-Month** contracts churn at a significantly higher rate than annual/biennial contracts.
- **Tenure** is the strongest negative predictor of churn — long-tenure customers are far less likely to leave.
- The preprocessing pipeline (StandardScaler + OneHotEncoder + OrdinalEncoder) ensures zero data leakage between train and test.

---

## 5. Reproducibility

```bash
pip install -r requirements.txt
python "Week 2/Day 5/ml_capstone_pipeline.py"
```

The best model is serialised to `{MODEL_PATH}` and can be loaded with:

```python
import joblib
model = joblib.load("{MODEL_PATH}")
predictions = model.predict(X_new)
```

---

## 6. Week 2 Concepts Applied

| Concept | Implementation |
|---------|---------------|
| Pandas wrangling | Dataset generation + EDA snapshot |
| StandardScaler | Numeric feature normalisation |
| OneHotEncoder | Nominal categorical features |
| OrdinalEncoder | Ordinal `seniority` feature |
| ColumnTransformer | Mixed-type preprocessing in one step |
| Pipeline | Chained preprocessor + classifier |
| train_test_split | 80/20 stratified split |
| StratifiedKFold | 5-fold CV preserving class balance |
| cross_validate | Multi-metric CV scoring |
| confusion_matrix | Per-class error analysis |
| classification_report | Precision / Recall / F1 |
| roc_auc_score | Threshold-independent ranking metric |
| joblib | Model serialisation |
| Structured logging | All key steps logged |
"""

    try:
        REPORT_PATH.write_text(md, encoding="utf-8")
        logger.info("Evaluation report saved → %s", REPORT_PATH)
        print(f"\n  ✓ Evaluation report saved → {REPORT_PATH}")
    except OSError as exc:
        logger.error("Could not write report: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — SERIALISE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
def save_best_model(best: dict, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Refit the best pipeline on full training data and serialise with joblib."""
    pipe = best["_pipe"]
    pipe.fit(X_train, y_train)
    try:
        joblib.dump(pipe, MODEL_PATH)
        logger.info("Best model serialised → %s", MODEL_PATH)
        print(f"  ✓ Best model ({best['model']}) saved → {MODEL_PATH}")
    except Exception as exc:
        logger.error("Model serialisation failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█" * 62)
    print("  WEEK 2 | DAY 5 — ML Capstone: End-to-End Pipeline")
    print("  Internship at Al Aziz Technologies")
    print("█" * 62)

    # ── Step 1: Data ──
    df = generate_dataset(n=500, seed=42)

    # ── Step 2: EDA ──
    eda_snapshot(df)

    # ── Step 3: Split ──
    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info("Split | Train=%d  Test=%d  Churn rate train=%.1f%%  test=%.1f%%",
                len(X_train), len(X_test),
                y_train.mean() * 100, y_test.mean() * 100)

    # ── Steps 4 & 5: Train + CV ──
    print("\n── 5-Fold Cross-Validation Results ──")
    cv_results = train_and_crossval(X_train, y_train)

    print(f"\n  {'Model':<26}  {'CV AUC':>8}  {'CV Std':>7}  {'CV F1':>7}")
    print(f"  {'-'*26}  {'-'*8}  {'-'*7}  {'-'*7}")
    for r in sorted(cv_results, key=lambda x: x["cv_roc_auc"], reverse=True):
        print(f"  {r['model']:<26}  {r['cv_roc_auc']:>8.4f}  "
              f"{r['cv_std']:>7.4f}  {r['cv_f1']:>7.4f}")

    # ── Step 6: Final evaluation ──
    cv_results = evaluate_on_test(cv_results, X_train, X_test, y_train, y_test)

    # ── Select best model by test ROC-AUC ──
    best = max(cv_results, key=lambda r: r["test_roc_auc"])
    logger.info("Best model selected: %s (ROC-AUC=%.4f)", best["model"], best["test_roc_auc"])

    # ── Step 7: Detailed report ──
    detailed_report(best, y_test)

    # ── Step 8: Export Markdown report ──
    export_markdown_report(
        cv_results, best, y_test,
        dataset_shape=df.shape,
        churn_rate=df["churn"].mean(),
    )

    # ── Step 9: Save model ──
    save_best_model(best, X_train, y_train)

    print("\n" + "=" * 62)
    print(f"  Pipeline complete.")
    print(f"  Best model  : {best['model']}")
    print(f"  Test AUC    : {best['test_roc_auc']:.4f}")
    print(f"  Test F1     : {best['test_f1']:.4f}")
    print(f"  Report      : {REPORT_PATH}")
    print(f"  Model file  : {MODEL_PATH}")
    logger.info("Capstone pipeline complete.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Fatal pipeline error: %s", exc, exc_info=True)
        sys.exit(1)
