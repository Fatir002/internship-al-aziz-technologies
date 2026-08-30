# 🤖 Week 2 | Day 4 — Scikit-Learn Model Training & Evaluation

> **Internship at Al Aziz Technologies · AI Engineering Track**

Trains and compares five classification models and two regression models on a customer churn dataset — covering the full Scikit-Learn estimator API, cross-validation, all key evaluation metrics, confusion matrix analysis, and an illustrated study of overfitting vs underfitting using Decision Tree depth.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | Scikit-Learn estimator API — `.fit()`, `.predict()`, `.predict_proba()` |
| 2 | Classification models — Logistic Regression, Decision Tree, Random Forest, KNN, SVM |
| 3 | Regression models — Linear Regression, Random Forest Regressor |
| 4 | Evaluation metrics — Accuracy, Precision, Recall, F1, ROC-AUC |
| 5 | Confusion matrix — TP, TN, FP, FN interpretation |
| 6 | `classification_report` — per-class breakdown |
| 7 | Regression metrics — MSE, RMSE, R² |
| 8 | 5-Fold Stratified Cross-Validation |
| 9 | Overfitting vs Underfitting — Decision Tree depth experiment |
| 10 | Model comparison table sorted by test ROC-AUC |

---

## 🚀 How to Run

```bash
pip install numpy pandas scikit-learn
python "Week 2/Day 4/model_training_eval.py"
```

---

## 🧩 Script Structure

```
model_training_eval.py
│
├── generate_churn_dataset()      # 400-row synthetic customer churn dataset
├── build_preprocessor()          # ColumnTransformer (StandardScaler + OHE + OrdinalEncoder)
│
├── run_classification()          # train 5 classifiers, CV + test eval, comparison table
├── run_regression()              # train 2 regressors, R² / RMSE comparison
└── demo_overfit_underfit()       # Decision Tree at 4 depths — train vs test accuracy gap
```

---

## 💡 Key Concepts Demonstrated

### Training Inside a Pipeline (Correct Pattern)
```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

pipe = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier())])

# 5-fold CV on training data only
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")

# Final evaluation on held-out test set
pipe.fit(X_train, y_train)
y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]
```

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

acc     = accuracy_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
```

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
```

### Overfitting vs Underfitting
```python
for depth in [1, 4, 8, None]:
    pipe = Pipeline([("pre", preprocessor),
                     ("clf", DecisionTreeClassifier(max_depth=depth))])
    pipe.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_acc  = accuracy_score(y_test,  pipe.predict(X_test))
    gap = train_acc - test_acc   # large gap = overfitting
```

---

## 📊 Sample Output

```
  Model Comparison (sorted by Test ROC-AUC):

  Model                   CV ROC-AUC  CV Std  Accuracy  Precision  Recall    F1    ROC-AUC
  Random Forest             0.7821    0.0312    0.7625     0.6341  0.5902  0.6114   0.7913
  Gradient Boosting         0.7734    0.0287    0.7500     0.6102  0.5738  0.5915   0.7842
  Logistic Regression       0.7601    0.0341    0.7375     0.5882  0.5574  0.5724   0.7698

── Overfitting vs Underfitting ──
  Config                   Train Acc   Test Acc    Gap  Verdict
  Depth=1  (Underfit)        0.7167     0.7000  0.0167  Underfitting
  Depth=4  (Balanced)        0.7667     0.7375  0.0292  Well-fitted
  Depth=8  (Good)            0.8262     0.7625  0.0637  Well-fitted
  Depth=None (Overfit)       1.0000     0.7375  0.2625  Overfitting
```

---

## 🛠️ Skills Demonstrated

`Scikit-Learn` · `Classification` · `Regression` · `Cross-Validation` · `ROC-AUC` · `Confusion Matrix` · `Overfitting Detection` · `Model Comparison` · `Pipeline` · `Evaluation Metrics`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
