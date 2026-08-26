# 🏗️ Week 1 | Day 3 — Object-Oriented Programming Practice

> **Internship at Al Aziz Technologies · AI Engineering Track**

A practical OOP project that builds a mini ML experiment tracker from scratch — demonstrating inheritance, encapsulation, properties, dunder methods, and composition through four progressively layered classes.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | Classes, `__init__`, instance vs class attributes |
| 2 | Instance methods, `@classmethod`, `@staticmethod` |
| 3 | Encapsulation with `_` (protected) and `__` (private) prefixes |
| 4 | `@property` and `@setter` for controlled attribute access |
| 5 | Inheritance and `super()` |
| 6 | Method overriding |
| 7 | Dunder methods: `__str__`, `__repr__`, `__len__`, `__iter__` |
| 8 | Composition — `ExperimentTracker` owns a list of `MLExperiment` objects |
| 9 | Practical project: ML experiment leaderboard |

---

## 🚀 How to Run

```bash
python "Week 1/Day 3/oop_practice.py"
```

> Requires **Python 3.10+**. No third-party packages — standard library only.

---

## 🧩 Class Architecture

```
DataRecord  (base)
│   ├── record_id, created_at
│   ├── _total_records  [class attribute]
│   ├── total_created() [classmethod]
│   └── validate_id()   [staticmethod]
│
├── Employee(DataRecord)
│   ├── name, department, _salary
│   ├── salary  [@property + @setter]
│   ├── annual_salary()
│   └── give_raise(percent)
│
└── MLExperiment(DataRecord)
    ├── model_name, hyperparams, metrics, _status
    ├── log_metrics(**metrics)
    ├── is_complete()
    └── summary()

ExperimentTracker  (composition — owns MLExperiment objects)
    ├── add(experiment)
    ├── best(metric)
    ├── print_leaderboard(metric)
    ├── __len__()
    └── __iter__()
```

---

## 💡 Key Concepts Demonstrated

### Class vs Instance Attributes
```python
class DataRecord:
    _total_records: int = 0          # class attribute — shared by all instances

    def __init__(self, record_id: int) -> None:
        self.record_id = record_id   # instance attribute — unique per object
        DataRecord._total_records += 1

print(DataRecord.total_created())    # classmethod → 3
```

### Encapsulation with `@property`
```python
class Employee(DataRecord):
    def __init__(self, ...):
        self._salary = salary        # protected by convention

    @property
    def salary(self) -> float:
        return self._salary

    @salary.setter
    def salary(self, amount: float) -> None:
        if amount < 0:
            raise ValueError("Salary cannot be negative")
        self._salary = amount
```

### Inheritance & `super()`
```python
class Employee(DataRecord):
    def __init__(self, record_id, name, department, salary):
        super().__init__(record_id)  # calls DataRecord.__init__
        self.name = name
        ...
```

### Dunder Methods
```python
def __str__(self) -> str:
    return f"Employee(id={self.record_id}, name={self.name!r}, ...)"

def __repr__(self) -> str:
    return f"Employee(id={self.record_id})"

def __len__(self) -> int:         # len(tracker)
    return len(self._experiments)

def __iter__(self):               # for exp in tracker
    return iter(self._experiments)
```

### Composition — ExperimentTracker
```python
tracker = ExperimentTracker("Customer Churn Prediction")

exp = MLExperiment(1, "RandomForest", {"n_estimators": 100})
exp.log_metrics(accuracy=0.9214, f1=0.9187, roc_auc=0.9631)
tracker.add(exp)

best = tracker.best(metric="roc_auc")
tracker.print_leaderboard(metric="accuracy")
```

---

## 📊 Sample Output

```
── Employee Class ──
  Employee(id=101, name='Fatir Faraz',  dept='AI Research',  salary=PKR 75,000)
  Employee(id=102, name='Ayesha Malik', dept='Data Science', salary=PKR 85,000)

  Fatir Faraz annual salary: PKR 900,000
  After 10% raise: PKR 82,500

── Leaderboard: Customer Churn Prediction (ranked by accuracy) ──
  #1  Exp #3 | GradientBoosting        | Status: complete  | accuracy=0.9352  |  f1=0.9301  |  roc_auc=0.9712
  #2  Exp #2 | RandomForestClassifier  | Status: complete  | accuracy=0.9214  |  f1=0.9187  |  roc_auc=0.9631
  #3  Exp #4 | SVM                     | Status: complete  | accuracy=0.8903  |  f1=0.8841  |  roc_auc=0.9388

  Best by ROC-AUC: GradientBoosting (0.9712)
```

---

## 🛠️ Skills Demonstrated

`Python OOP` · `Inheritance & Composition` · `Encapsulation` · `Design Patterns` · `ML Experiment Tracking` · `Type Hints` · `Dunder Methods`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
