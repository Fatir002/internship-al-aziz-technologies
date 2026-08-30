# 📊 Week 2 | Day 2 — Exploratory Data Analysis & Data Visualisation

> **Internship at Al Aziz Technologies · AI Engineering Track**

A full EDA workflow on a synthetic employee dataset — computing descriptive statistics, detecting outliers with the IQR method, building a Pearson correlation matrix, and generating six publication-quality plots with Matplotlib saved to disk. No GUI or display required.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | Descriptive statistics — mean, median, std, variance, skewness |
| 2 | Pearson correlation matrix |
| 3 | Outlier detection — IQR method (Q1, Q3, fence bounds) |
| 4 | Histogram with mean & median lines |
| 5 | Bar chart — average salary by department with value labels |
| 6 | Scatter plot — experience vs salary coloured by department + trend line |
| 7 | Line chart — average performance score by experience bucket |
| 8 | Correlation heatmap with annotated values |
| 9 | Box plot — salary distribution by department |

---

## 🚀 How to Run

```bash
pip install numpy pandas matplotlib
python "Week 2/Day 2/eda_and_visualization.py"
```

Plots are saved automatically to your system temp directory. To view them interactively:

```bash
SHOW_PLOTS=1 python "Week 2/Day 2/eda_and_visualization.py"
```

---

## 🧩 Script Structure

```
eda_and_visualization.py
│
├── generate_dataset()           # 50-row synthetic employee dataset
├── eda_descriptive()            # describe(), variance, skewness, correlation matrix
├── eda_outliers()               # IQR method for salary, score, experience
│
├── plot_histogram()             # salary / score / experience with mean+median lines
├── plot_bar_chart()             # avg salary by dept with bar labels
├── plot_scatter()               # experience vs salary, coloured by dept + trend line
├── plot_line_chart()            # avg score by experience bucket (pd.cut)
├── plot_heatmap()               # annotated Pearson correlation heatmap
└── plot_boxplot()               # salary spread by department
```

---

## 💡 Key Concepts Demonstrated

### Outlier Detection — IQR Method
```python
q1, q3 = df["salary"].quantile(0.25), df["salary"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["salary"] < lower) | (df["salary"] > upper)]
```

### Pearson Correlation Matrix
```python
numeric = df.select_dtypes(include="number").drop(columns=["id"])
corr = numeric.corr()   # Pearson by default
```

### Scatter Plot with Trend Line
```python
m, b = np.polyfit(df["experience_years"], df["salary"], 1)
x_line = np.linspace(df["experience_years"].min(), df["experience_years"].max(), 100)
ax.plot(x_line, m * x_line + b, linestyle="--", label=f"Trend (r={corr_val:.2f})")
```

### Binning with `pd.cut`
```python
df["exp_bin"] = pd.cut(
    df["experience_years"],
    bins=[0, 3, 6, 9, 15],
    labels=["1-3 yrs", "4-6 yrs", "7-9 yrs", "10+ yrs"]
)
avg_score = df.groupby("exp_bin", observed=True)["performance_score"].mean()
```

### Saving Plots (No GUI Required)
```python
import matplotlib
matplotlib.use("Agg")   # headless — works in any environment

fig.savefig("plot.png", dpi=120, bbox_inches="tight")
```

---

## 📁 Plots Generated

| File | Description |
|------|-------------|
| `01_histograms.png` | Salary, performance score, experience — mean & median lines |
| `02_bar_salary_by_dept.png` | Average salary per department with value labels |
| `03_scatter_exp_salary.png` | Experience vs salary coloured by department + trend |
| `04_line_score_by_exp.png` | Average performance score by experience bucket |
| `05_correlation_heatmap.png` | Annotated Pearson correlation heatmap |
| `06_boxplot_salary_by_dept.png` | Salary distribution spread by department |

---

## 📊 Sample EDA Output

```
── Descriptive Statistics ──
                     count    mean      std  variance    min     50%      max  skewness
experience_years      50.0    8.16     4.16     17.30   1.00    8.00    15.00     -0.04
salary                50.0  89308.0  16842.0  283K     53000  88500  128000       0.23
performance_score     50.0   80.38     9.64     92.92  62.00   80.05   100.0       0.07

── Outlier Detection (IQR Method) ──
  salary                Q1=75000  Q3=103000  IQR=28000  bounds=[33000, 145000]  outliers=0
  performance_score     Q1=72.8   Q3=88.6    IQR=15.8   bounds=[48.9, 112.5]   outliers=0
```

---

## 🛠️ Skills Demonstrated

`EDA` · `Matplotlib` · `Statistical Analysis` · `Outlier Detection` · `Correlation Analysis` · `Data Visualisation` · `Pandas` · `NumPy` · `Feature Distribution Analysis`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
