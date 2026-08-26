# 📦 Week 1 | Day 2 — Python Data Structures & Control Flow

> **Internship at Al Aziz Technologies · AI Engineering Track**

A comprehensive hands-on walkthrough of Python's four core data structures and control flow tools, framed around real AI/ML use cases — from storing model metrics in nested dictionaries to filtering datasets with `break` and `continue`.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | Lists — creation, indexing, slicing, mutation, methods |
| 2 | Tuples — immutability, unpacking, use as fixed records |
| 3 | Sets — union, intersection, difference, deduplication |
| 4 | Dictionaries — keys/values, nested dicts, model metrics |
| 5 | Control Flow — `if/elif/else`, `for`, `while`, `break`, `continue` |
| 6 | List & Dictionary Comprehensions |
| 7 | Nested Data Structures — list-of-dicts employee dataset |

---

## 🚀 How to Run

```bash
python "Week 1/Day 2/data_structures.py"
```

> Requires **Python 3.10+**. No third-party packages — standard library only.

---

## 🧩 Script Structure

```
data_structures.py
│
├── demo_lists()                 # AI frameworks list, scores, slicing, sort
├── demo_tuples()                # Employee record, immutability, model configs
├── demo_sets()                  # Week topics, set algebra, deduplication
├── demo_dictionaries()          # Intern profile, nested model metrics table
├── demo_control_flow()          # Grade classifier, epoch loop, data filter
├── demo_comprehensions()        # Squares, evens, normalised, 4×4 table
└── demo_nested_structures()     # Employee dataset — filter, sort, aggregate
```

---

## 💡 Key Concepts Demonstrated

### Lists
```python
frameworks = ["NumPy", "Pandas", "PyTorch", "TensorFlow"]
frameworks.append("LangChain")
frameworks.sort()
print(frameworks[2:5])   # slicing
```

### Tuples — Immutable Records
```python
employee = (1001, "Fatir Faraz", "AI Engineer", 85_000.0)
emp_id, name, role, salary = employee   # unpacking
```

### Sets — Deduplication & Algebra
```python
week1 = {"Python", "NumPy", "Pandas"}
week2 = {"Pandas", "Scikit-Learn", "NumPy"}
print(week1 & week2)   # intersection → {'NumPy', 'Pandas'}
print(week1 | week2)   # union
```

### Nested Dictionary — Model Metrics
```python
model_metrics = {
    "RandomForest": {"accuracy": 0.92, "f1": 0.91, "roc_auc": 0.96},
    "SVM":          {"accuracy": 0.89, "f1": 0.88, "roc_auc": 0.93},
}
```

### List Comprehension
```python
squares    = [n ** 2 for n in range(1, 11)]
evens      = [n for n in range(1, 11) if n % 2 == 0]
normalised = [round(n / 10, 2) for n in range(1, 11)]
```

### Control Flow — Training Loop
```python
epoch, loss = 1, 1.0
while loss > 0.1:
    loss *= 0.65
    epoch += 1
    if epoch > 20:
        break
```

---

## 📊 Sample Output Highlights

```
── Model Metrics ──
Model                  Accuracy     F1  ROC-AUC
---------------------- --------- ------ --------
LogisticRegression         0.870  0.850    0.910
RandomForest               0.920  0.910    0.960
SVM                        0.890  0.880    0.930

── Salary Ranking ──
1. Ayesha     (AI Research    ) PKR 95,000
2. Chand      (AI Research    ) PKR 91,000
3. Ehsan      (Data Science   ) PKR 88,000
```

---

## 🛠️ Skills Demonstrated

`Python` · `Data Structures` · `List Comprehensions` · `Control Flow` · `Data Filtering & Aggregation` · `ML Metric Tracking`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
