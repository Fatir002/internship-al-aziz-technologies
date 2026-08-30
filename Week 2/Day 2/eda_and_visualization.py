"""
Week 2 | Day 2 — EDA & Data Visualisation
==========================================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - Exploratory Data Analysis (EDA) workflow
    - Descriptive statistics: mean, median, mode, variance, std
    - Correlation matrix (Pearson)
    - Outlier detection with IQR method
    - Matplotlib: histograms, bar charts, scatter plots, line charts
    - Seaborn: heatmap (correlation), boxplot, pairplot (subset)
    - Saving plots to temp directory (no GUI required)

Usage:
    python "Week 2/Day 2/eda_and_visualization.py"

    To display plots interactively (requires GUI backend):
        SHOW_PLOTS=1 python "Week 2/Day 2/eda_and_visualization.py"
"""

import logging
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works in any environment
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("week2.day2")

SHOW = os.environ.get("SHOW_PLOTS", "0") == "1"
PLOT_DIR = Path(tempfile.gettempdir()) / "alaziz_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATASET — 50 employee records for richer EDA
# ══════════════════════════════════════════════════════════════════════════════
def generate_dataset(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic employee dataset with realistic distributions."""
    rng = np.random.default_rng(seed)
    departments = ["AI Research", "Data Science", "Engineering", "Product"]
    dept_array = rng.choice(departments, size=n, p=[0.3, 0.3, 0.25, 0.15])
    exp = rng.integers(1, 16, size=n).astype(float)

    # Salary correlated with experience + dept premium
    base_salary = 55_000 + exp * 4_000 + rng.normal(0, 5_000, size=n)
    dept_premium = np.where(dept_array == "AI Research", 10_000,
                   np.where(dept_array == "Data Science", 5_000, 0))
    salary = np.clip(base_salary + dept_premium, 50_000, 130_000).round(-2)

    # Score correlated with exp with some noise
    score = np.clip(60 + exp * 2.5 + rng.normal(0, 5, size=n), 60, 100).round(1)

    df = pd.DataFrame({
        "id":                range(1, n + 1),
        "department":        dept_array,
        "experience_years":  exp,
        "salary":            salary,
        "performance_score": score,
        "tenure_years":      rng.integers(1, 10, size=n),
        "projects_completed":rng.integers(2, 25, size=n),
    })
    logger.info("Dataset generated — shape: %s", df.shape)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
def eda_descriptive(df: pd.DataFrame) -> None:
    logger.info("── Descriptive Statistics ──")
    print("\n── Descriptive Statistics ──")
    numeric = df.select_dtypes(include="number").drop(columns=["id"])
    desc = numeric.describe().T
    desc["variance"] = numeric.var()
    desc["skewness"] = numeric.skew()
    print(f"\n{desc[['count','mean','std','variance','min','50%','max','skewness']].round(2).to_string()}\n")

    # Mode per department
    mode_dept = df["department"].mode()[0]
    print(f"  Most common department: {mode_dept}")

    # Correlation
    print("\n── Pearson Correlation Matrix ──")
    corr = numeric.corr()
    print(corr.round(3).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 2. OUTLIER DETECTION — IQR Method
# ══════════════════════════════════════════════════════════════════════════════
def eda_outliers(df: pd.DataFrame) -> None:
    logger.info("── Outlier Detection (IQR) ──")
    print("\n── Outlier Detection (IQR Method) ──")
    for col in ["salary", "performance_score", "experience_years"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"  {col:<22}  Q1={q1:.0f}  Q3={q3:.0f}  IQR={iqr:.0f}  "
              f"bounds=[{lower:.0f}, {upper:.0f}]  outliers={len(outliers)}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig: plt.Figure, name: str) -> None:
    path = PLOT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    logger.info("Plot saved → %s", path)
    if SHOW:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 3. HISTOGRAM — Salary Distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_histogram(df: pd.DataFrame) -> None:
    logger.info("── Histogram ──")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Week 2 | Day 2 — Histograms", fontsize=14, fontweight="bold")

    cols = ["salary", "performance_score", "experience_years"]
    for ax, col, color in zip(axes, cols, PALETTE[:3]):
        ax.hist(df[col], bins=12, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(df[col].mean(), color="black", linestyle="--", linewidth=1.2, label=f"mean={df[col].mean():.0f}")
        ax.axvline(df[col].median(), color="red", linestyle=":", linewidth=1.2, label=f"median={df[col].median():.0f}")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "01_histograms.png")
    print(f"  Histogram saved to: {PLOT_DIR / '01_histograms.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BAR CHART — Avg Salary by Department
# ══════════════════════════════════════════════════════════════════════════════
def plot_bar_chart(df: pd.DataFrame) -> None:
    logger.info("── Bar Chart ──")
    dept_avg = df.groupby("department")["salary"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(dept_avg.index, dept_avg.values, color=PALETTE[:len(dept_avg)], edgecolor="white")
    ax.bar_label(bars, labels=[f"PKR {v:,.0f}" for v in dept_avg.values], padding=4, fontsize=9)
    ax.set_title("Average Salary by Department", fontsize=13, fontweight="bold")
    ax.set_xlabel("Department")
    ax.set_ylabel("Average Salary (PKR)")
    ax.set_ylim(0, dept_avg.max() * 1.15)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    _save(fig, "02_bar_salary_by_dept.png")
    print(f"  Bar chart saved to: {PLOT_DIR / '02_bar_salary_by_dept.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCATTER PLOT — Experience vs Salary (coloured by dept)
# ══════════════════════════════════════════════════════════════════════════════
def plot_scatter(df: pd.DataFrame) -> None:
    logger.info("── Scatter Plot ──")
    fig, ax = plt.subplots(figsize=(9, 6))
    departments = df["department"].unique()
    for dept, color in zip(departments, PALETTE):
        mask = df["department"] == dept
        ax.scatter(df.loc[mask, "experience_years"], df.loc[mask, "salary"],
                   label=dept, color=color, alpha=0.75, s=60, edgecolors="white", linewidth=0.5)

    # Trend line
    m, b = np.polyfit(df["experience_years"], df["salary"], 1)
    x_line = np.linspace(df["experience_years"].min(), df["experience_years"].max(), 100)
    ax.plot(x_line, m * x_line + b, color="black", linestyle="--", linewidth=1.5, label=f"Trend (r={df['experience_years'].corr(df['salary']):.2f})")

    ax.set_title("Experience vs Salary (coloured by Department)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experience (years)")
    ax.set_ylabel("Salary (PKR)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "03_scatter_exp_salary.png")
    print(f"  Scatter saved to: {PLOT_DIR / '03_scatter_exp_salary.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. LINE CHART — Avg Score by Experience bin
# ══════════════════════════════════════════════════════════════════════════════
def plot_line_chart(df: pd.DataFrame) -> None:
    logger.info("── Line Chart ──")
    df2 = df.copy()
    df2["exp_bin"] = pd.cut(df2["experience_years"], bins=[0, 3, 6, 9, 15],
                             labels=["1-3 yrs", "4-6 yrs", "7-9 yrs", "10+ yrs"])
    avg_score = df2.groupby("exp_bin", observed=True)["performance_score"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(avg_score.index.astype(str), avg_score.values, marker="o", color=PALETTE[0],
            linewidth=2.5, markersize=8, markerfacecolor="white", markeredgewidth=2)
    for x, y in zip(avg_score.index.astype(str), avg_score.values):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    ax.set_title("Average Performance Score by Experience Level", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experience Bucket")
    ax.set_ylabel("Avg Performance Score")
    ax.set_ylim(avg_score.min() - 5, avg_score.max() + 8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, "04_line_score_by_exp.png")
    print(f"  Line chart saved to: {PLOT_DIR / '04_line_score_by_exp.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. HEATMAP — Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════════
def plot_heatmap(df: pd.DataFrame) -> None:
    logger.info("── Heatmap ──")
    numeric = df.select_dtypes(include="number").drop(columns=["id"])
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black" if abs(val) < 0.7 else "white", fontsize=8)

    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_correlation_heatmap.png")
    print(f"  Heatmap saved to: {PLOT_DIR / '05_correlation_heatmap.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. BOX PLOT — Salary by Department
# ══════════════════════════════════════════════════════════════════════════════
def plot_boxplot(df: pd.DataFrame) -> None:
    logger.info("── Box Plot ──")
    departments = df["department"].unique()
    data_by_dept = [df.loc[df["department"] == d, "salary"].values for d in departments]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data_by_dept, labels=departments, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Salary Distribution by Department (Box Plot)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Department")
    ax.set_ylabel("Salary (PKR)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save(fig, "06_boxplot_salary_by_dept.png")
    print(f"  Box plot saved to: {PLOT_DIR / '06_boxplot_salary_by_dept.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█" * 60)
    print("  WEEK 2 | DAY 2 — EDA & Data Visualisation")
    print("  Internship at Al Aziz Technologies")
    print("█" * 60)

    df = generate_dataset(n=50, seed=42)

    eda_descriptive(df)
    eda_outliers(df)

    print(f"\n── Generating Plots (saved to {PLOT_DIR}) ──")
    plot_histogram(df)
    plot_bar_chart(df)
    plot_scatter(df)
    plot_line_chart(df)
    plot_heatmap(df)
    plot_boxplot(df)

    print(f"\n  All 6 plots saved to: {PLOT_DIR}")
    print("  Set SHOW_PLOTS=1 to display interactively.")
    print("\n" + "=" * 60)
    logger.info("Day 2 — EDA & Visualisation complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
