# ⚙️ Week 1 | Day 3 — Functions, Modules & Scope

> **Internship at Al Aziz Technologies · AI Engineering Track**

A deep dive into writing reusable, well-structured Python functions — covering everything from basic definitions and default arguments to `*args`, `**kwargs`, lambda expressions, closures, and a functional data-preprocessing pipeline typical of real AI workflows.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | Defining functions with parameters, defaults, type hints, and docstrings |
| 2 | Return values and multiple returns (`tuple` unpacking) |
| 3 | `*args` — variable positional arguments |
| 4 | `**kwargs` — variable keyword arguments |
| 5 | Lambda functions with `map`, `filter`, and `reduce` |
| 6 | Variable scope — the **LEGB rule** (Local → Enclosing → Global → Built-in) |
| 7 | Closures and factory functions |
| 8 | Standard library modules: `math`, `random`, `datetime`, `os` |
| 9 | Functional pipeline: `map → filter → reduce` for data preprocessing |

---

## 🚀 How to Run

```bash
python "Week 1/Day 3/modules_and_functions.py"
```

> Requires **Python 3.10+**. No third-party packages — standard library only.

---

## 🧩 Script Structure

```
modules_and_functions.py
│
├── greet()                      # Basic function, default parameter
├── celsius_to_fahrenheit()      # Simple conversion, type hints
├── divide()                     # Safe division, returns None on ZeroDivision
├── minmax()                     # Multiple return values via tuple
│
├── summarise_scores(*scores)    # *args → stats dict (count, mean, std…)
├── build_model_config(**kwargs) # **kwargs → model config dict
│
├── demo_lambdas()               # map, filter, reduce, max with key
├── demo_scope()                 # LEGB + closure (threshold checker factory)
├── demo_stdlib_modules()        # math, random, datetime, os
└── pipeline_demo()              # parse → filter → aggregate pipeline
```

---

## 💡 Key Concepts Demonstrated

### Default Parameters & Multiple Returns
```python
def greet(name: str, role: str = "AI Engineer") -> str:
    return f"Hello, {name}! Welcome to the {role} internship."

def minmax(numbers: list[float]) -> tuple[float, float]:
    return min(numbers), max(numbers)

lo, hi = minmax([4.2, 1.1, 9.8])   # tuple unpacking
```

### `*args` — Variable Positional Arguments
```python
def summarise_scores(*scores: float) -> dict[str, float]:
    mean = sum(scores) / len(scores)
    ...

stats = summarise_scores(88, 72, 95, 61, 84)
```

### `**kwargs` — Variable Keyword Arguments
```python
def build_model_config(model_name: str, **hyperparams) -> dict:
    return {"model": model_name, "params": hyperparams}

cfg = build_model_config("RandomForest", n_estimators=200, max_depth=10)
```

### Lambda + `map` / `filter` / `reduce`
```python
from functools import reduce

square  = lambda x: x ** 2
numbers = [3, 7, 1, 9, 4]

squared = list(map(square, numbers))
evens   = list(filter(lambda n: n % 2 == 0, numbers))
product = reduce(lambda a, b: a * b, numbers)
```

### LEGB Scope & Closures
```python
GLOBAL_THRESHOLD = 0.5   # Global scope

def make_threshold_checker(threshold: float):
    def check(value: float) -> bool:
        return value > threshold   # 'threshold' lives in enclosing scope
    return check

above_half = make_threshold_checker(0.5)
print(above_half(0.72))   # True
```

### Functional Preprocessing Pipeline
```python
raw = ["88", "NaN", "72", "N/A", "95"]

# Step 1 — parse
parsed = list(map(safe_parse, raw))           # [88.0, None, 72.0, None, 95.0]

# Step 2 — filter
valid = [v for v in parsed if v is not None]  # [88.0, 72.0, 95.0]

# Step 3 — aggregate
mean = reduce(lambda a, b: a + b, valid) / len(valid)
```

---

## 📊 Sample Output Highlights

```
── Score Summary (*args) ──
  count   : 7.0
  mean    : 81.0
  min     : 61.0
  max     : 95.0
  std     : 10.8167

── Closure Threshold=0.5 Checks ──
  ✗ p=0.30 → below
  ✓ p=0.55 → above
  ✓ p=0.72 → above

── Functional Pipeline ──
  Raw     : ['88', 'NaN', '72', '95', 'N/A', '61']
  Valid   : [88.0, 72.0, 95.0, 61.0]
  Mean    : 79.00
```

---

## 🛠️ Skills Demonstrated

`Python` · `Functional Programming` · `Type Hints` · `Closures & Scope` · `Data Preprocessing` · `Standard Library` · `Clean Code Patterns`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
