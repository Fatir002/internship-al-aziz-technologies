# 🐍 Week 1 | Day 1 — Python Fundamentals

> **Internship at Al Aziz Technologies · AI Engineering Track**

This script covers the absolute foundations of Python with an AI/ML context — from variables and data types all the way to error handling and beginner exercises. Every concept is demonstrated with practical, runnable examples.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | Why Python dominates AI development |
| 2 | Variables and dynamic typing |
| 3 | Core data types: `int`, `float`, `str`, `bool`, `NoneType` |
| 4 | Type conversion — explicit (casting) and implicit |
| 5 | Arithmetic, comparison, logical, and bitwise operators |
| 6 | String operations and f-strings |
| 7 | Input / Output and print formatting |
| 8 | Basic `try` / `except` / `else` / `finally` error handling |
| 9 | Exercises: Temperature Converter, Simple Interest, FizzBuzz |

---

## 🚀 How to Run

```bash
python "Week 1/Day 1/fundamentals.py"
```

> Requires **Python 3.10+** (uses `match`-compatible type hints). No third-party packages needed — standard library only.

---

## 🧩 Script Structure

```
fundamentals.py
│
├── why_python_for_ai()          # 5 reasons Python leads in AI
├── demo_variables_and_types()   # int, float, str, bool, None
├── demo_type_conversion()       # str→float→int→str, bool casting
├── demo_operators()             # arithmetic, comparison, logical, bitwise
├── demo_strings()               # upper/lower/replace/split/slice/f-strings
├── demo_input_output()          # formatted print tables (input mocked)
├── demo_error_handling()        # ValueError, ZeroDivisionError
└── exercises()                  # 3 beginner practice problems
```

---

## 💡 Key Concepts Demonstrated

### Dynamic Typing
```python
x = 100          # x is int
x = "now a str"  # legal — Python rebinds the name
```

### Type Conversion
```python
pi_str   = "3.14159"
pi_float = float(pi_str)   # → 3.14159
pi_int   = int(pi_float)   # → 3  (truncates, not rounds)
```

### F-strings
```python
name = "Al Aziz Technologies"
print(f"Welcome to {name}!")
```

### Error Handling
```python
try:
    result = int("hello")
except ValueError as exc:
    print(f"Could not convert: {exc}")
finally:
    pass  # cleanup runs regardless
```

---

## 📝 Exercises

| Exercise | Description |
|----------|-------------|
| 1 | Celsius → Fahrenheit converter for `[0, 20, 37, 100]°C` |
| 2 | Simple interest calculator (`P=10000, R=8%, T=3 years`) |
| 3 | FizzBuzz from 1 to 20 |

---

## 🗂️ Part of the Series

| Day | File | Topic |
|-----|------|-------|
| **Day 1** | `fundamentals.py` | Python Fundamentals ← *you are here* |
| Day 2 | `data_structures.py` | Data Structures & Control Flow |
| Day 3 | `modules_and_functions.py` | Functions, Modules & Scope |
| Day 3 | `oop_practice.py` | Object-Oriented Programming |

---

## 👤 Author

**Fatir Faraz** — AI Engineering Intern, Al Aziz Technologies  
GitHub: [@Fatir002](https://github.com/Fatir002)
