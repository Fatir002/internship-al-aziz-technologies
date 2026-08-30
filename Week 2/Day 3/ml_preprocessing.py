"""
Week 2 | Day 3 — Machine Learning Fundamentals & Preprocessing
===============================================================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - What is Machine Learning? (AI vs ML vs Deep Learning)
    - Supervised vs Unsupervised learning
    - Classification, Regression, Clustering
    - Features and labels
    - Train / Test / Validation split
    - Data preprocessing:
        * Min-Max Scaling (MinMaxScaler)
        * Z-Score Standardisation (StandardScaler)
        * One-Hot Encoding (pd.get_dummies / OneHotEncoder)
        * Label Encoding (LabelEncoder)
        * Ordinal Encoding
    - Scikit-Learn Pipeline construction
    - ColumnTransformer for mixed-type data

Run:
    python "Week 2/Day 3/ml_preprocessing.py"
"""

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("week2.day3")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATASET — customer churn (300 rows)
# ══════════════════════════════════════════════════════════════════════════════
def generate_churn_dataset(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic customer churn dataset."""
    rng = np.random.default_rng(seed)
    age         = rng.integers(18, 70,  size=n)
    tenure      = rng.integers(1,  120, size=n)
    monthly_fee = rng.uniform(20, 150,  size=n).round(2)
    contracts   = rng.choice(["Month-to-Month", "One Year", "Two Year"], size=n, p=[0.5, 0.3, 0.2])
    internet    = rng.choice(["DSL", "Fiber", "No"], size=n, p=[0.4, 0.4, 0.2])
    support     = rng.choice(["Yes", "No"], size=n, p=[0.4, 0.6])
    seniority   = rng.choice(["Junior", "Mid", "Senior"], size=n, p=[0.3, 0.4, 0.3])

    # Churn probability influenced by contract & tenure
    churn_prob  = (
        0.35
        - 0.2  * (contracts == "Two Year")
        - 0.1  * (contracts == "One Year")
        - 0.003 * tenure
        + 0.15 * (internet == "Fiber")
        + rng.uniform(-0.1, 0.1, size=n)
    )
    churn = (rng.random(size=n) < np.clip(churn_prob, 0.05, 0.95)).astype(int)

    df = pd.DataFrame({
        "age":           age,
        "tenure_months": tenure,
        "monthly_fee":   monthly_fee,
        "contract":      contracts,
        "internet":      internet,
        "tech_support":  support,
        "seniority":     seniority,
        "churn":         churn,
    })
    logger.info("Churn dataset generated: shape=%s, churn_rate=%.1f%%",
                df.shape, df["churn"].mean() * 100)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. ML CONCEPTS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def explain_ml_concepts() -> None:
    print("\n" + "=" * 60)
    print("  ML CONCEPTS OVERVIEW")
    print("=" * 60)

    concepts = {
        "AI":             "Broad field — machines that simulate intelligence",
        "ML":             "Subset of AI — systems that learn from data",
        "Deep Learning":  "Subset of ML — multi-layer neural networks",
        "Supervised":     "Labelled data → learn input→output mapping",
        "Unsupervised":   "Unlabelled data → find hidden structure",
        "Classification": "Predict a discrete category (churn: Yes/No)",
        "Regression":     "Predict a continuous value (salary: PKR 85,000)",
        "Clustering":     "Group similar points (customer segments)",
        "Feature":        "Input variable (age, tenure, monthly_fee ...)",
        "Label / Target": "Output to predict (churn)",
    }
    for term, desc in concepts.items():
        print(f"  {term:<18} — {desc}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def demo_train_test_split(df: pd.DataFrame) -> tuple:
    """Split features and labels into train/validation/test sets."""
    logger.info("── Train/Test Split ──")
    print("\n── Train / Test Split ──")

    X = df.drop(columns=["churn"])
    y = df["churn"]

    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"  Full dataset  : {len(df):>5} rows  churn_rate={y.mean():.3f}")
    print(f"  Train set     : {len(X_train):>5} rows  churn_rate={y_train.mean():.3f}")
    print(f"  Validation set: {len(X_val):>5} rows  churn_rate={y_val.mean():.3f}")
    print(f"  Test set      : {len(X_test):>5} rows  churn_rate={y_test.mean():.3f}")
    print("  ✓ Stratification maintained — class ratios preserved across splits.")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 3. SCALING
# ══════════════════════════════════════════════════════════════════════════════
def demo_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    logger.info("── Feature Scaling ──")
    print("\n── Feature Scaling ──")
    numeric_cols = ["age", "tenure_months", "monthly_fee"]

    raw = X_train[numeric_cols].head(4).values

    # Min-Max
    mm = MinMaxScaler()
    mm.fit(X_train[numeric_cols])
    mm_transformed = mm.transform(X_train[numeric_cols])[:4]

    # Standard Scaler
    ss = StandardScaler()
    ss.fit(X_train[numeric_cols])
    ss_transformed = ss.transform(X_train[numeric_cols])[:4]

    print(f"  Numeric columns: {numeric_cols}")
    print(f"\n  Raw (first 4 rows):")
    for row in raw:
        print(f"    {row}")

    print(f"\n  MinMaxScaler → [0, 1]:")
    for row in mm_transformed:
        print(f"    {row.round(4)}")

    print(f"\n  StandardScaler → mean=0, std=1:")
    for row in ss_transformed:
        print(f"    {row.round(4)}")

    print(f"\n  Important: Fit on TRAIN only, transform both train and test.")
    print(f"  MinMaxScaler scale_  : {mm.scale_.round(6)}")
    print(f"  StandardScaler mean_ : {ss.mean_.round(2)}")
    print(f"  StandardScaler std_  : {ss.scale_.round(2)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENCODING
# ══════════════════════════════════════════════════════════════════════════════
def demo_encoding(df: pd.DataFrame) -> None:
    logger.info("── Categorical Encoding ──")
    print("\n── Categorical Encoding ──")

    sample = df[["contract", "internet", "tech_support", "seniority", "churn"]].head(6).copy()
    print(f"  Original:\n{sample}\n")

    # One-Hot Encoding
    ohe = OneHotEncoder(sparse_output=False, drop="first")
    ohe_result = ohe.fit_transform(sample[["contract", "internet"]])
    ohe_cols = ohe.get_feature_names_out(["contract", "internet"])
    ohe_df = pd.DataFrame(ohe_result.astype(int), columns=ohe_cols)
    print(f"  OneHotEncoder (contract, internet):\n{ohe_df}\n")

    # Label Encoding (target)
    le = LabelEncoder()
    le_result = le.fit_transform(sample["tech_support"])
    print(f"  LabelEncoder (tech_support): {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"  Encoded: {le_result.tolist()}\n")

    # Ordinal Encoding
    oe = OrdinalEncoder(categories=[["Junior", "Mid", "Senior"]])
    oe_result = oe.fit_transform(sample[["seniority"]])
    print(f"  OrdinalEncoder (seniority — Junior=0, Mid=1, Senior=2):")
    for orig, enc in zip(sample["seniority"], oe_result.flatten()):
        print(f"    {orig:<8} → {int(enc)}")

    # pd.get_dummies (quick alternative)
    dummy_df = pd.get_dummies(sample[["contract"]], drop_first=True)
    print(f"\n  pd.get_dummies (contract, drop_first=True):\n{dummy_df}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SKLEARN PIPELINE + COLUMNTRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════
def demo_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    logger.info("── Sklearn Pipeline + ColumnTransformer ──")
    print("\n── Scikit-Learn Pipeline ──")

    numeric_features = ["age", "tenure_months", "monthly_fee"]
    categorical_features = ["contract", "internet", "tech_support"]
    ordinal_features = ["seniority"]

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal_transformer = Pipeline([
        ("oe", OrdinalEncoder(categories=[["Junior", "Mid", "Senior"]])),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num",  numeric_transformer,     numeric_features),
        ("cat",  categorical_transformer, categorical_features),
        ("ord",  ordinal_transformer,     ordinal_features),
    ], remainder="drop")

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    ohe_cols = preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(categorical_features)
    all_cols  = numeric_features + list(ohe_cols) + ordinal_features

    print(f"  Input  features: {X_train.shape[1]}")
    print(f"  Output features: {X_train_proc.shape[1]}  (after OHE expansion)")
    print(f"  Feature names  : {all_cols}")
    print(f"\n  Preprocessed X_train (first 3 rows):")
    for row in X_train_proc[:3]:
        print(f"    {row.round(3)}")
    print(f"\n  Preprocessed X_test shape : {X_test_proc.shape}")
    print("  ✓ Pipeline fitted on train, applied to test (no data leakage).")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█" * 60)
    print("  WEEK 2 | DAY 3 — ML Fundamentals & Preprocessing")
    print("  Internship at Al Aziz Technologies")
    print("█" * 60)

    explain_ml_concepts()

    df = generate_churn_dataset(n=300, seed=42)
    print(f"\n  Dataset shape : {df.shape}")
    print(f"  Churn rate    : {df['churn'].mean():.1%}")
    print(f"\n  Sample:\n{df.head(4).to_string(index=False)}\n")

    X_train, X_val, X_test, y_train, y_val, y_test = demo_train_test_split(df)
    demo_scaling(X_train, X_test)
    demo_encoding(df)
    demo_pipeline(X_train, X_test)

    print("\n" + "=" * 60)
    logger.info("Day 3 — ML Preprocessing complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
