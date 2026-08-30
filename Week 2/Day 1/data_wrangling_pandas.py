"""
Week 2 | Day 1 — NumPy (deep dive) & Pandas Data Wrangling
===========================================================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - NumPy array shapes, broadcasting, mathematical ops (recap + deeper)
    - Pandas Series and DataFrames from scratch
    - Reading CSV / JSON from StringIO (inline, no file dependencies)
    - Selecting columns, filtering rows, sorting
    - Handling missing values (detect, drop, fill strategies)
    - Removing duplicates
    - Feature engineering (derived columns)
    - GroupBy aggregation
    - Merging / joining DataFrames
    - Exporting cleaned data to CSV string

Run:
    python "Week 2/Day 1/data_wrangling_pandas.py"
"""

import logging
import sys
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("week2.day1")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC CSV DATA (inline — no file required)
# ══════════════════════════════════════════════════════════════════════════════
RAW_CSV = """\
id,name,department,salary,experience_years,performance_score,joined_year,active
1,Ayesha Malik,AI Research,95000,6,91,2018,True
2,Bilal Khan,Data Science,82000,4,85,2020,True
3,Chand Rehman,AI Research,91000,5,88,2019,True
4,Dania Siddiqui,Engineering,78000,3,80,2021,True
5,Ehsan Tariq,Data Science,88000,5,87,2019,True
6,Fatima Noor,Engineering,72000,2,76,2022,True
7,Ghazal Butt,AI Research,105000,9,95,2015,True
8,Hamza Ali,Data Science,79000,3,81,2021,True
9,Iqra Shahid,Engineering,68000,1,73,2023,True
10,Jawad Mehmood,AI Research,98000,7,93,2017,True
11,Kashif Raza,Data Science,,4,82,2020,True
12,Layla Ahmed,Engineering,71000,,78,2022,True
13,Bilal Khan,Data Science,82000,4,85,2020,True
14,Maryam Shah,AI Research,NaN,8,90,2016,False
15,Noman Butt,Engineering,65000,1,,2023,False
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. NUMPY DEEP DIVE (shape, axis-ops, broadcasting)
# ══════════════════════════════════════════════════════════════════════════════
def demo_numpy_advanced() -> None:
    logger.info("── NumPy Advanced ──")
    print("\n── NumPy: Axis Operations & Broadcasting ──")

    rng = np.random.default_rng(42)
    # Simulate a (samples × features) data matrix
    X = rng.integers(50, 100, size=(6, 4)).astype(float)
    feature_names = ["Age", "Salary_k", "Experience", "Score"]
    print(f"  X (6 samples × 4 features):\n{X}")

    print(f"\n  Column means (axis=0): {X.mean(axis=0).round(2)}")
    print(f"  Row means    (axis=1): {X.mean(axis=1).round(2)}")
    print(f"  Column std   (axis=0): {X.std(axis=0).round(2)}")

    # Z-score normalisation with broadcasting
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma
    print(f"\n  Z-normalised (each col mean≈0, std≈1):\n{X_norm.round(3)}")

    # Masking
    high_score_mask = X[:, 3] > 80
    print(f"\n  Rows with Score > 80: indices {np.where(high_score_mask)[0].tolist()}")
    print(f"  Filtered:\n{X[high_score_mask]}")

    # Stack and split
    extra_col = rng.integers(1, 5, size=(6, 1)).astype(float)
    X_augmented = np.hstack([X, extra_col])
    print(f"\n  hstack with extra column → shape {X_augmented.shape}")
    first_half, second_half = np.split(X, 2, axis=0)
    print(f"  vsplit halves: {first_half.shape} / {second_half.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PANDAS SERIES
# ══════════════════════════════════════════════════════════════════════════════
def demo_series() -> None:
    logger.info("── Pandas Series ──")
    print("\n── Pandas Series ──")

    salaries = pd.Series(
        [95000, 82000, 91000, 78000, 88000],
        index=["Ayesha", "Bilal", "Chand", "Dania", "Ehsan"],
        name="salary",
    )
    print(f"\n  Series:\n{salaries}")
    print(f"\n  mean  : {salaries.mean():,.0f}")
    print(f"  median: {salaries.median():,.0f}")
    print(f"  std   : {salaries.std():,.0f}")

    # Boolean filter on Series
    high_earners = salaries[salaries > 85_000]
    print(f"\n  High earners (> 85k): {high_earners.to_dict()}")

    # Apply
    normalised = salaries.apply(lambda v: round((v - salaries.min()) / (salaries.max() - salaries.min()), 4))
    print(f"\n  Min-max normalised:\n{normalised}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATAFRAME CREATION & INSPECTION
# ══════════════════════════════════════════════════════════════════════════════
def load_dataframe() -> pd.DataFrame:
    """Load the synthetic CSV into a DataFrame and return it."""
    df = pd.read_csv(StringIO(RAW_CSV))
    logger.info("DataFrame loaded — shape: %s", df.shape)
    return df


def demo_dataframe_basics(df: pd.DataFrame) -> None:
    logger.info("── DataFrame Basics ──")
    print("\n── DataFrame Inspection ──")
    print(f"  shape    : {df.shape}")
    print(f"  columns  : {df.columns.tolist()}")
    print(f"  dtypes   :\n{df.dtypes.to_string()}")
    print(f"\n  head(5):\n{df.head()}\n")
    print(f"  describe():\n{df.describe().round(2)}\n")
    print(f"  info():")
    df.info()


# ══════════════════════════════════════════════════════════════════════════════
# 4. SELECTING, FILTERING, SORTING
# ══════════════════════════════════════════════════════════════════════════════
def demo_select_filter_sort(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("── Select / Filter / Sort ──")
    print("\n── Selecting & Filtering ──")

    # Column selection
    subset = df[["name", "department", "salary", "performance_score"]]
    print(f"  Selected columns:\n{subset.head(3)}\n")

    # Row filter
    ai_team = df[df["department"] == "AI Research"]
    print(f"  AI Research team ({len(ai_team)} rows):\n{ai_team[['name','salary','performance_score']]}\n")

    # Multi-condition filter
    senior_high = df[(df["experience_years"] >= 5) & (df["salary"] > 85_000)]
    print(f"  Senior high earners (exp≥5, salary>85k):\n{senior_high[['name','experience_years','salary']]}\n")

    # Sorting
    sorted_df = df.sort_values(["department", "salary"], ascending=[True, False])
    print(f"  Sorted by dept then salary desc:\n{sorted_df[['name','department','salary']].head(6)}\n")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
def demo_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("── Missing Values ──")
    print("\n── Missing Values ──")

    print(f"  Null counts:\n{df.isnull().sum()}\n")
    print(f"  Total nulls: {df.isnull().sum().sum()}")

    # Strategy 1: fill numeric with median
    df["salary"] = df["salary"].fillna(df["salary"].median())
    df["experience_years"] = df["experience_years"].fillna(df["experience_years"].median())

    # Strategy 2: fill categorical with mode
    df["performance_score"] = df["performance_score"].fillna(df["performance_score"].mode()[0])

    print(f"\n  After imputation — null counts:\n{df.isnull().sum()}\n")
    logger.info("Missing values handled (median/mode imputation).")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. DUPLICATES
# ══════════════════════════════════════════════════════════════════════════════
def demo_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("── Duplicates ──")
    print("\n── Duplicate Rows ──")
    n_dupes = df.duplicated(subset=["name", "department", "salary"]).sum()
    print(f"  Duplicate rows detected: {n_dupes}")
    df = df.drop_duplicates(subset=["name", "department", "salary"], keep="first")
    df = df.reset_index(drop=True)
    print(f"  Shape after dedup: {df.shape}")
    logger.info("Duplicates removed. Clean shape: %s", df.shape)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def demo_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("── Feature Engineering ──")
    print("\n── Feature Engineering ──")

    # Annual salary
    df["annual_salary"] = df["salary"] * 12

    # Salary per year of experience
    df["salary_per_year_exp"] = (df["salary"] / df["experience_years"].replace(0, 1)).round(2)

    # Tenure from joined year
    CURRENT_YEAR = 2025
    df["tenure_years"] = CURRENT_YEAR - df["joined_year"]

    # Performance band
    def perf_band(score: float) -> str:
        if score >= 90: return "Excellent"
        elif score >= 80: return "Good"
        elif score >= 70: return "Satisfactory"
        return "Needs Improvement"

    df["perf_band"] = df["performance_score"].apply(perf_band)

    # Seniority flag
    df["is_senior"] = (df["experience_years"] >= 5).astype(int)

    print(f"  New columns added: annual_salary, salary_per_year_exp, tenure_years, perf_band, is_senior")
    print(f"\n{df[['name','annual_salary','salary_per_year_exp','tenure_years','perf_band','is_senior']].to_string(index=False)}\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. GROUPBY AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
def demo_groupby(df: pd.DataFrame) -> None:
    logger.info("── GroupBy Aggregation ──")
    print("\n── GroupBy Aggregation ──")

    dept_stats = df.groupby("department").agg(
        headcount=("id", "count"),
        avg_salary=("salary", "mean"),
        max_salary=("salary", "max"),
        avg_score=("performance_score", "mean"),
        avg_tenure=("tenure_years", "mean"),
    ).round(2)

    print(f"  Department stats:\n{dept_stats}\n")

    band_counts = df.groupby("perf_band")["name"].count().sort_values(ascending=False)
    print(f"  Performance band distribution:\n{band_counts}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9. MERGE (JOIN)
# ══════════════════════════════════════════════════════════════════════════════
def demo_merge(df: pd.DataFrame) -> None:
    logger.info("── Merge / Join ──")
    print("\n── Merge (Join) ──")

    dept_info = pd.DataFrame({
        "department": ["AI Research", "Data Science", "Engineering"],
        "dept_head":  ["Dr. Zara Khan", "Prof. Asim Ali", "Eng. Sara Naz"],
        "budget_PKR": [15_000_000, 12_000_000, 10_000_000],
    })
    print(f"  dept_info table:\n{dept_info}\n")

    merged = df.merge(dept_info, on="department", how="left")
    print(f"  After left join on 'department' — shape: {merged.shape}")
    print(f"\n{merged[['name','department','salary','dept_head','budget_PKR']].head(6).to_string(index=False)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█" * 60)
    print("  WEEK 2 | DAY 1 — NumPy Deep Dive & Pandas Data Wrangling")
    print("  Internship at Al Aziz Technologies")
    print("█" * 60)

    demo_numpy_advanced()
    demo_series()

    df = load_dataframe()
    demo_dataframe_basics(df)
    df = demo_select_filter_sort(df)
    df = demo_missing_values(df)
    df = demo_duplicates(df)
    df = demo_feature_engineering(df)
    demo_groupby(df)
    demo_merge(df)

    print("\n" + "=" * 60)
    logger.info("Day 1 — Data Wrangling complete. Clean shape: %s", df.shape)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
