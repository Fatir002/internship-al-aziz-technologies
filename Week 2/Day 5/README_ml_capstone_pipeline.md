# 🚀 Week 2 Capstone — End-to-End ML Pipeline: Customer Churn Prediction

> **Internship at Al Aziz Technologies · AI Engineering Track**

A production-style, self-contained machine learning pipeline that takes raw data from generation all the way to a serialised, deployable model — implementing every Week 2 concept in a single orchestrated script. Trains 5 classifiers with 5-fold stratified cross-validation, evaluates on a held-out test set, exports a Markdown evaluation report, and saves the best model with `joblib`.

---

## ✨ What This Pipeline Does

| Step | Description |
|------|-------------|
| 1 | Generate a 500-row synthetic customer churn dataset |
| 2 | EDA snapshot — shape, churn rate, null check, feature distributions |
| 3 | Build a `ColumnTransformer` preprocessing pipeline (3 feature types) |
| 4 | Train 5 classifiers wrapped in Scikit-Learn `Pipeline` objects |
| 5 | 5-fold Stratified Cross-Validation with 5 scoring metrics |
| 6 | Final evaluation on a 20% held-out test set |
| 7 | Detailed confusion matrix + `classification_report` for best model |
| 8 | Export full evaluation report as `evaluation_report.md` |
| 9 | Serialise best model to `/tmp/best_model.joblib` |

---

## 🚀 How to Run

```bash
pip install numpy pandas scikit-learn joblib
python "Week 2/Day 5/ml_capstone_pipeline.py"
```

### Load the Saved Model
```python
import joblib
model = joblib.load("/tmp/best_model.joblib")
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

---

## 🧩 Pipeline Architecture

```
ml_capstone_pipeline.py
│
├── generate_dataset(n=500)          # synthetic churn data with realistic correlations
├── eda_snapshot()                   # shape, churn rate, nulls, value counts, churn by contract
│
├── build_preprocessor()             # ColumnTransformer:
│   ├── StandardScaler               #   numeric: age, tenure_months, monthly_fee
│   ├── OneHotEncoder                #   nominal: contract, internet, tech_support
│   └── OrdinalEncoder               #   ordinal: seniority (Junior < Mid < Senior)
│
├── train_and_crossval()             # 5 models × 5-fold CV × 5 scoring metrics
│   ├── Logistic Regression
│   ├── Random Forest
│   ├── Gradient Boosting
│   ├── SVM (RBF)
│   └── K-Nearest Neighbours
│
├── evaluate_on_test()               # refit on full train, score on held-out test
├── detailed_report()                # confusion matrix + classification_report
├── export_markdown_report()         # writes evaluation_report.md
└── save_best_model()                # joblib.dump → /tmp/best_model.joblib
```

---

## 💡 Key Implementation Highlights

### Multi-Metric Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold, cross_validate

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "f1", "roc_auc", "precision", "recall"]

cv_result = cross_validate(pipe, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)

cv_roc_auc = cv_result["test_roc_auc"].mean()
cv_std      = cv_result["test_roc_auc"].std()
```

### ColumnTransformer — Mixed Feature Types
```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),
     ["age", "tenure_months", "monthly_fee"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     ["contract", "internet", "tech_support"]),
    ("ord", OrdinalEncoder(categories=[["Junior", "Mid", "Senior"]]),
     ["seniority"]),
], remainder="drop")
```

### Model Serialisation with joblib
```python
import joblib

joblib.dump(best_pipeline, "/tmp/best_model.joblib")

# Load anywhere
model = joblib.load("/tmp/best_model.joblib")
preds = model.predict(X_new)
```

### Automatic Best Model Selection
```python
best = max(cv_results, key=lambda r: r["test_roc_auc"])
```

---

## 📊 Sample Output

```
── 5-Fold Cross-Validation Results ──
  Model                       CV AUC   CV Std    CV F1
  Gradient Boosting           0.7891   0.0271   0.5923
  Random Forest               0.7834   0.0308   0.5841
  Logistic Regression         0.7712   0.0334   0.5688
  SVM (RBF)                   0.7645   0.0291   0.5512
  K-Nearest Neighbours        0.7203   0.0412   0.5134

── Best Model: Gradient Boosting ──
  Test ROC-AUC : 0.7956
  Test F1      : 0.6012
  Test Accuracy: 0.7600

  Confusion Matrix:
             Pred 0   Pred 1
  Actual 0 :    62        8
  Actual 1 :    16       14

  ✓ Evaluation report saved → Week 2/Day 5/evaluation_report.md
  ✓ Best model (Gradient Boosting) saved → /tmp/best_model.joblib
```

---

## 📄 Output Files

| File | Description |
|------|-------------|
| `evaluation_report.md` | Full Markdown report — dataset summary, model comparison table, confusion matrix, key findings |
| `/tmp/best_model.joblib` | Serialised best pipeline ready for inference |

---

## 🛠️ Skills Demonstrated

`End-to-End ML Pipeline` · `Scikit-Learn` · `ColumnTransformer` · `Pipeline` · `Cross-Validation` · `Gradient Boosting` · `Random Forest` · `ROC-AUC` · `Confusion Matrix` · `Model Serialisation` · `joblib` · `Structured Logging` · `Markdown Report Generation`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
