# 🐼 Week 2 | Day 1 — NumPy Deep Dive & Pandas Data Wrangling

> **Internship at Al Aziz Technologies · AI Engineering Track**

A complete data wrangling workflow covering the full pre-modelling pipeline — from loading raw messy data to a clean, feature-engineered DataFrame ready for ML. Demonstrates every core Pandas operation used in real data science and AI engineering roles.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | NumPy axis operations, Z-score normalisation via broadcasting, masking, `hstack`/`vsplit` |
| 2 | Pandas `Series` — indexing, boolean filtering, `.apply()` |
| 3 | DataFrame creation from CSV, `.info()`, `.describe()`, dtype inspection |
| 4 | Column selection, row filtering (single & multi-condition), sorting |
| 5 | Missing value detection (`isnull`), median/mode imputation |
| 6 | Duplicate detection and removal |
| 7 | Feature engineering — derived columns, binning, boolean flags |
| 8 | GroupBy aggregation — headcount, avg salary, avg score per department |
| 9 | DataFrame merging — left join on department key |

---

## 🚀 How to Run

```bash
pip install numpy pandas
python "Week 2/Day 1/data_wrangling_pandas.py"
```

> No external files needed — the dataset is embedded as an inline CSV string.

---

## 🧩 Script Structure

```
data_wrangling_pandas.py
│
├── demo_numpy_advanced()        # axis ops, broadcasting, masking, hstack/vsplit
├── demo_series()                # Series ops, boolean filter, apply + lambda
├── load_dataframe()             # read inline CSV via StringIO
├── demo_dataframe_basics()      # shape, dtypes, head, describe, info
├── demo_select_filter_sort()    # column select, row filter, multi-condition, sort
├── demo_missing_values()        # isnull, fillna (median/mode imputation)
├── demo_duplicates()            # duplicated, drop_duplicates, reset_index
├── demo_feature_engineering()   # annual_salary, salary_per_exp, tenure, perf_band
├── demo_groupby()               # groupby.agg — headcount, avg salary, avg score
└── demo_merge()                 # left join employee data with dept_info table
```

---

## 💡 Key Concepts Demonstrated

### Missing Value Imputation
```python
# Numeric — fill with median (robust to outliers)
df["salary"] = df["salary"].fillna(df["salary"].median())

# Categorical — fill with mode
df["performance_score"] = df["performance_score"].fillna(
    df["performance_score"].mode()[0]
)
```

### Feature Engineering
```python
df["annual_salary"]       = df["salary"] * 12
df["salary_per_year_exp"] = (df["salary"] / df["experience_years"]).round(2)
df["tenure_years"]        = 2025 - df["joined_year"]
df["is_senior"]           = (df["experience_years"] >= 5).astype(int)

def perf_band(score):
    if score >= 90: return "Excellent"
    elif score >= 80: return "Good"
    return "Satisfactory"

df["perf_band"] = df["performance_score"].apply(perf_band)
```

### GroupBy Aggregation
```python
dept_stats = df.groupby("department").agg(
    headcount    = ("id",                "count"),
    avg_salary   = ("salary",            "mean"),
    max_salary   = ("salary",            "max"),
    avg_score    = ("performance_score", "mean"),
    avg_tenure   = ("tenure_years",      "mean"),
).round(2)
```

### DataFrame Merge (Left Join)
```python
merged = df.merge(dept_info, on="department", how="left")
```

### NumPy Broadcasting (Z-Score)
```python
mu    = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma == 0] = 1.0          # avoid division by zero
X_norm = (X - mu) / sigma        # broadcasts across all rows
```

---

## 📊 Sample Output

```
── Department Stats (GroupBy) ──
              headcount  avg_salary  max_salary  avg_score  avg_tenure
AI Research           4    97250.00      105000      91.75        7.25
Data Science          3    83000.00       88000      84.33        4.33
Engineering           4    71750.00       78000      76.75        2.75

── After Imputation ──
  Null counts: all zeros — clean dataset ready for ML
```

---

## 🛠️ Skills Demonstrated

`Pandas` · `NumPy` · `Data Wrangling` · `Missing Value Imputation` · `Feature Engineering` · `GroupBy Aggregation` · `DataFrame Merging` · `Data Cleaning`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
