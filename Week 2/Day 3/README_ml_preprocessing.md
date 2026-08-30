# ⚙️ Week 2 | Day 3 — Machine Learning Fundamentals & Preprocessing

> **Internship at Al Aziz Technologies · AI Engineering Track**

A structured introduction to machine learning concepts followed by a hands-on implementation of every major preprocessing technique used in production ML pipelines — scaling, encoding, and a full Scikit-Learn `Pipeline` + `ColumnTransformer` for mixed-type data.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | AI vs ML vs Deep Learning — concept map |
| 2 | Supervised vs Unsupervised, Classification vs Regression vs Clustering |
| 3 | Features, labels, and the train/val/test split — stratified 70/15/15 |
| 4 | `MinMaxScaler` — scales to [0, 1] |
| 5 | `StandardScaler` — Z-score standardisation (mean=0, std=1) |
| 6 | `OneHotEncoder` — nominal categorical features |
| 7 | `LabelEncoder` — binary categorical target |
| 8 | `OrdinalEncoder` — ordered categorical features |
| 9 | `pd.get_dummies` — quick one-hot alternative |
| 10 | `ColumnTransformer` — applies different transformers to different column types |
| 11 | Scikit-Learn `Pipeline` — chains preprocessor + classifier, prevents data leakage |

---

## 🚀 How to Run

```bash
pip install numpy pandas scikit-learn
python "Week 2/Day 3/ml_preprocessing.py"
```

> Dataset is generated synthetically — no external files required.

---

## 🧩 Script Structure

```
ml_preprocessing.py
│
├── generate_churn_dataset()     # 300-row synthetic customer churn dataset
├── explain_ml_concepts()        # printed ML concept reference card
├── demo_train_test_split()      # stratified 70/15/15 split with class ratio checks
├── demo_scaling()               # MinMaxScaler vs StandardScaler comparison
├── demo_encoding()              # OHE, LabelEncoder, OrdinalEncoder, get_dummies
└── demo_pipeline()              # ColumnTransformer + Pipeline (no data leakage)
```

---

## 💡 Key Concepts Demonstrated

### Stratified Train/Val/Test Split
```python
from sklearn.model_selection import train_test_split

# 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
# Split temp 50/50 → 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
```

### Scaling — Fit on Train Only (No Leakage)
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

mm = MinMaxScaler()
mm.fit(X_train[numeric_cols])         # learn from train only
X_train_scaled = mm.transform(X_train[numeric_cols])
X_test_scaled  = mm.transform(X_test[numeric_cols])  # apply same transform
```

### OneHotEncoder + OrdinalEncoder
```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ohe = OneHotEncoder(sparse_output=False, drop="first")
ohe_result = ohe.fit_transform(df[["contract", "internet"]])

oe = OrdinalEncoder(categories=[["Junior", "Mid", "Senior"]])
oe_result = oe.fit_transform(df[["seniority"]])   # Junior=0, Mid=1, Senior=2
```

### ColumnTransformer + Pipeline (Production Pattern)
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),      ["age", "tenure_months", "monthly_fee"]),
    ("cat", OneHotEncoder(...),    ["contract", "internet", "tech_support"]),
    ("ord", OrdinalEncoder(...),   ["seniority"]),
], remainder="drop")

pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", SomeClassifier()),
])
pipe.fit(X_train, y_train)   # preprocessor fits on train, transforms test — zero leakage
```

---

## 📊 Sample Output

```
── Train / Test Split ──
  Full dataset  :   300 rows  churn_rate=0.274
  Train set     :   210 rows  churn_rate=0.276
  Validation set:    45 rows  churn_rate=0.267
  Test set      :    45 rows  churn_rate=0.267
  ✓ Stratification maintained — class ratios preserved across splits.

── Feature Scaling (first 4 rows) ──
  Raw:           [29.0, 67.0, 89.4]  [45.0, 12.0, 34.7]
  MinMaxScaler:  [0.216, 0.567, 0.48] [0.529, 0.093, 0.12]
  StandardScaler: [-1.01, 0.43, 0.76] [0.88, -1.32, -1.42]
```

---

## 🛠️ Skills Demonstrated

`Scikit-Learn` · `ML Preprocessing` · `Feature Scaling` · `Categorical Encoding` · `Pipeline Design` · `ColumnTransformer` · `Data Leakage Prevention` · `Stratified Splitting`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
