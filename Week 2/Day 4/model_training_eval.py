"""
Week 2 | Day 4 — Scikit-Learn Model Training & Evaluation
==========================================================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - Introduction to scikit-learn estimator API (.fit / .predict / .score)
    - Classification models:
        * Logistic Regression
        * Decision Tree Classifier
        * Random Forest Classifier
        * K-Nearest Neighbours
        * Support Vector Machine (SVM)
    - Regression models:
        * Linear Regression
        * Random Forest Regressor
    - Evaluation metrics:
        * Accuracy, Precision, Recall, F1-Score
        * Confusion Matrix
        * ROC-AUC
        * Mean Squared Error (MSE), Root MSE, R²
    - Cross-validation (5-fold, StratifiedKFold)
    - Overfitting vs Underfitting (illustrated with train vs val scores)
    - Model comparison table

Run:
    python "Week 2/Day 4/model_training_eval.py"
"""

import logging
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("week2.day4")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATOR (reused from Day 3)
# ══════════════════════════════════════════════════════════════════════════════
def generate_churn_dataset(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age         = rng.integers(18, 70, size=n)
    tenure      = rng.integers(1, 120, size=n)
    monthly_fee = rng.uniform(20, 150, size=n).round(2)
    contracts   = rng.choice(["Month-to-Month", "One Year", "Two Year"], size=n, p=[0.5, 0.3, 0.2])
    internet    = rng.choice(["DSL", "Fiber", "No"], size=n, p=[0.4, 0.4, 0.2])
    support     = rng.choice(["Yes", "No"], size=n, p=[0.4, 0.6])
    seniority   = rng.choice(["Junior", "Mid", "Senior"], size=n, p=[0.3, 0.4, 0.3])

    churn_prob = (
        0.35
        - 0.20 * (contracts == "Two Year")
        - 0.10 * (contracts == "One Year")
        - 0.003 * tenure
        + 0.15 * (internet == "Fiber")
        + rng.uniform(-0.1, 0.1, size=n)
    )
    churn = (rng.random(size=n) < np.clip(churn_prob, 0.05, 0.95)).astype(int)

    return pd.DataFrame({
        "age": age, "tenure_months": tenure, "monthly_fee": monthly_fee,
        "contract": contracts, "internet": internet,
        "tech_support": support, "seniority": seniority, "churn": churn,
    })


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), ["age", "tenure_months", "monthly_fee"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["contract", "internet", "tech_support"]),
        ("ord", OrdinalEncoder(categories=[["Junior", "Mid", "Senior"]]),
         ["seniority"]),
    ], remainder="drop")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CLASSIFICATION — Train Multiple Models & Compare
# ══════════════════════════════════════════════════════════════════════════════
def run_classification(df: pd.DataFrame) -> None:
    logger.info("── Classification: Training & Evaluation ──")
    print("\n" + "═" * 62)
    print("  SECTION 1 — CLASSIFICATION (Customer Churn)")
    print("═" * 62)

    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info("Train: %d  |  Test: %d  |  Churn rate: %.1f%%",
                len(X_train), len(X_test), y.mean() * 100)

    preprocessor = build_preprocessor()

    classifiers: list[tuple[str, object]] = [
        ("Logistic Regression",   LogisticRegression(max_iter=1000, random_state=42)),
        ("Decision Tree",         DecisionTreeClassifier(max_depth=6, random_state=42)),
        ("Random Forest",         RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)),
        ("K-Nearest Neighbours",  KNeighborsClassifier(n_neighbors=7)),
        ("SVM (RBF)",             SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
    ]

    results: list[dict] = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in classifiers:
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

        # 5-Fold CV on train
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf,
                                    scoring="roc_auc", n_jobs=-1)

        # Fit on full train, evaluate on test
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        prec    = precision_score(y_test, y_pred, zero_division=0)
        rec     = recall_score(y_test, y_pred, zero_division=0)
        f1      = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model":       name,
            "CV ROC-AUC":  round(cv_scores.mean(), 4),
            "CV Std":      round(cv_scores.std(),  4),
            "Accuracy":    round(acc,  4),
            "Precision":   round(prec, 4),
            "Recall":      round(rec,  4),
            "F1":          round(f1,   4),
            "ROC-AUC":     round(roc_auc, 4),
        })
        logger.info("%-24s  CV AUC=%.4f±%.4f  Test AUC=%.4f  F1=%.4f",
                    name, cv_scores.mean(), cv_scores.std(), roc_auc, f1)

    # ── Comparison table ──
    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    print(f"\n  Model Comparison (sorted by Test ROC-AUC):\n")
    print(results_df.to_string(index=False))

    # ── Best model: detailed report ──
    best_name = results_df.iloc[0]["Model"]
    best_clf  = dict(classifiers)[best_name]
    best_pipe = Pipeline([("pre", build_preprocessor()), ("clf", best_clf)])
    best_pipe.fit(X_train, y_train)
    y_pred_best = best_pipe.predict(X_test)

    print(f"\n  ── Best Model: {best_name} ──")
    print(f"\n  Confusion Matrix:\n")
    cm = confusion_matrix(y_test, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    print(f"  {'':>10} Pred 0   Pred 1")
    print(f"  Actual 0 :  {tn:>5}    {fp:>5}")
    print(f"  Actual 1 :  {fn:>5}    {tp:>5}")
    print(f"\n  Classification Report:")
    report_lines = classification_report(
        y_test, y_pred_best, target_names=["No Churn", "Churn"]
    )
    for line in report_lines.splitlines():
        print("  " + line)


# ══════════════════════════════════════════════════════════════════════════════
# 2. REGRESSION — Predict Monthly Fee
# ══════════════════════════════════════════════════════════════════════════════
def run_regression(df: pd.DataFrame) -> None:
    logger.info("── Regression: Training & Evaluation ──")
    print("\n" + "═" * 62)
    print("  SECTION 2 — REGRESSION (Predict Monthly Fee)")
    print("═" * 62)

    X = df.drop(columns=["monthly_fee", "churn"])
    y = df["monthly_fee"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    reg_preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), ["age", "tenure_months"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["contract", "internet", "tech_support"]),
        ("ord", OrdinalEncoder(categories=[["Junior", "Mid", "Senior"]]),
         ["seniority"]),
    ], remainder="drop")

    regressors: list[tuple[str, object]] = [
        ("Linear Regression",      LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)),
    ]

    print(f"\n  {'Model':<26}  {'Train R²':>9}  {'Test R²':>8}  {'RMSE':>8}  {'MSE':>10}")
    print(f"  {'-'*26}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*10}")

    for name, reg in regressors:
        pipe = Pipeline([("pre", reg_preprocessor), ("reg", reg)])
        pipe.fit(X_train, y_train)

        y_pred_tr = pipe.predict(X_train)
        y_pred_te = pipe.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_tr)
        test_r2  = r2_score(y_test,  y_pred_te)
        mse      = mean_squared_error(y_test, y_pred_te)
        rmse     = np.sqrt(mse)

        print(f"  {name:<26}  {train_r2:>9.4f}  {test_r2:>8.4f}  {rmse:>8.2f}  {mse:>10.2f}")
        logger.info("%-26s  Train R²=%.4f  Test R²=%.4f  RMSE=%.2f", name, train_r2, test_r2, rmse)

    print(f"\n  Note: High train R² + low test R² = overfitting.")
    print(f"  Note: Both scores low = underfitting.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. OVERFITTING VS UNDERFITTING ILLUSTRATION
# ══════════════════════════════════════════════════════════════════════════════
def demo_overfit_underfit(df: pd.DataFrame) -> None:
    logger.info("── Overfitting vs Underfitting Illustration ──")
    print("\n" + "═" * 62)
    print("  SECTION 3 — OVERFITTING vs UNDERFITTING")
    print("═" * 62)

    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preprocessor = build_preprocessor()

    depth_configs = [
        ("Depth=1  (Underfit)", 1),
        ("Depth=4  (Balanced)", 4),
        ("Depth=8  (Good)",     8),
        ("Depth=None (Overfit)", None),
    ]

    print(f"\n  {'Config':<26}  {'Train Acc':>10}  {'Test Acc':>9}  {'Gap':>8}  {'Verdict'}")
    print(f"  {'-'*26}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*15}")

    for label, depth in depth_configs:
        pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", DecisionTreeClassifier(max_depth=depth, random_state=42)),
        ])
        pipe.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, pipe.predict(X_train))
        te_acc = accuracy_score(y_test,  pipe.predict(X_test))
        gap    = tr_acc - te_acc
        verdict = (
            "Underfitting" if te_acc < 0.72 else
            "Overfitting"  if gap > 0.12 else
            "Well-fitted"
        )
        print(f"  {label:<26}  {tr_acc:>10.4f}  {te_acc:>9.4f}  {gap:>8.4f}  {verdict}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█" * 62)
    print("  WEEK 2 | DAY 4 — Model Training & Evaluation")
    print("  Internship at Al Aziz Technologies")
    print("█" * 62)

    df = generate_churn_dataset(n=400, seed=42)
    logger.info("Dataset ready: shape=%s, churn_rate=%.1f%%",
                df.shape, df["churn"].mean() * 100)

    run_classification(df)
    run_regression(df)
    demo_overfit_underfit(df)

    print("\n" + "=" * 62)
    logger.info("Day 4 — Model Training & Evaluation complete.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
