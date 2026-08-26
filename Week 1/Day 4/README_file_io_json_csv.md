# 📁 Week 1 | Day 4 — File I/O, JSON, CSV & Logging

> **Internship at Al Aziz Technologies · AI Engineering Track**

Covers the data persistence and observability fundamentals every AI/ML engineer needs daily — reading and writing text files, serialising ML experiment results as JSON, processing employee records with CSV, managing secrets through environment variables, and configuring production-grade rotating log handlers.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | Plain text file I/O — `open()`, `write`, `read`, `append` modes |
| 2 | CSV I/O — `csv.DictWriter`, `csv.DictReader`, in-memory `StringIO` |
| 3 | JSON — `json.dumps`, `json.load`, serialising ML experiment results |
| 4 | Environment variables — `os.environ`, secrets management, `.env` pattern |
| 5 | Structured logging — console + `RotatingFileHandler`, all five log levels |
| 6 | Exception handling at every I/O boundary — `OSError`, `JSONDecodeError` |

---

## 🚀 How to Run

```bash
python "Week 1/Day 4/file_io_json_csv.py"
```

> Standard library only — no third-party packages required. All files are written to the system temp directory; no cleanup needed.

---

## 🧩 Script Structure

```
file_io_json_csv.py
│
├── demo_text_file_io()           # write / read / append a formatted text report
├── demo_csv_io()                 # DictWriter → StringIO → DictReader + filter
├── demo_json_io()                # serialise experiment results, load & inspect
├── demo_environment_variables()  # os.environ, secrets masking, .env pattern
└── demo_logging_levels()         # DEBUG / INFO / WARNING / ERROR / CRITICAL
```

---

## 💡 Key Concepts Demonstrated

### Text File I/O
```python
# Write
with open("report.txt", "w", encoding="utf-8") as fh:
    fh.write("Al Aziz Technologies — Report\n")

# Read
with open("report.txt", "r", encoding="utf-8") as fh:
    content = fh.read()

# Append
with open("report.txt", "a", encoding="utf-8") as fh:
    fh.write("Footer line\n")
```

### CSV with DictWriter / DictReader
```python
import csv
from io import StringIO

buffer = StringIO()
writer = csv.DictWriter(buffer, fieldnames=["id", "name", "salary"])
writer.writeheader()
writer.writerows(EMPLOYEES)

buffer.seek(0)
reader = csv.DictReader(buffer)
active = [r for r in reader if r["active"] == "True"]
```

### JSON — Serialising ML Results
```python
import json

payload = {
    "project": "Customer Churn Prediction",
    "best_model": "GradientBoosting",
    "experiments": EXPERIMENTS,
}

# Serialise
json_str = json.dumps(payload, indent=2)

# Load back
with open("experiments.json", "r") as fh:
    data = json.load(fh)
```

### Rotating Log Handler (Production Pattern)
```python
import logging.handlers

handler = logging.handlers.RotatingFileHandler(
    "app.log", maxBytes=500_000, backupCount=2, encoding="utf-8"
)
logging.basicConfig(handlers=[logging.StreamHandler(), handler])
```

### Environment Variables & Secrets
```python
import os

api_key     = os.environ.get("API_KEY", "NOT_SET")   # from .env in real projects
max_retries = int(os.environ.get("MAX_RETRIES", "3"))

# Never hardcode secrets — use python-dotenv + .env file
```

---

## 📊 Sample Output

```
── CSV Read-back ──
  ID   Name                   Dept             Salary   Active
  ---- ---------------------- ---------------- -------- ------
  1    Ayesha Malik           AI Research        95,000  True
  2    Bilal Khan             Data Science       82,000  True
  3    Chand Rehman           AI Research        91,000  False

  Active employees (4): ['Ayesha Malik', 'Bilal Khan', 'Dania Siddiqui', 'Ehsan Tariq']

── JSON Read-back ──
  Project     : Customer Churn Prediction
  Best model  : GradientBoosting
  Experiments : 5
  [exp_001] LogisticRegression     acc=0.8721  f1=0.8534
  [exp_003] GradientBoosting       acc=0.9352  f1=0.9301
```

---

## 🛠️ Skills Demonstrated

`Python` · `File I/O` · `JSON Serialisation` · `CSV Processing` · `Structured Logging` · `Secrets Management` · `Production Error Handling` · `Data Pipelines`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
