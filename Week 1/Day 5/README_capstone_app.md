# 🚀 Week 1 Capstone — CLI Analytics Tool

> **Internship at Al Aziz Technologies · AI Engineering Track**

A fully functional command-line analytics application built as the Week 1 capstone project — integrating every concept from the internship's first week into one cohesive, production-style Python tool. Analyses an employee dataset with NumPy-powered statistics, fetches live weather from a REST API, and generates a formatted analytics report — all routed through a clean `argparse` CLI.

---

## ✨ Features

- **Employee analytics** — descriptive stats (mean, median, std, min, max), department breakdown, top performers, Pearson correlation
- **Live weather** — real-time data from Open-Meteo API (free, no API key) for any city by coordinates
- **Full report** — combined analytics + weather report generated as formatted text
- **CLI interface** — `argparse` subcommands with `--help` and custom flags
- **Production patterns** — dataclasses, structured logging, typed interfaces, robust error handling at every I/O boundary

---

## 🚀 How to Run

```bash
pip install numpy requests
```

```bash
# Analyse the employee dataset
python capstone_app.py stats

# Fetch live weather for Islamabad (default)
python capstone_app.py weather

# Fetch weather for a custom city
python capstone_app.py weather --city Tokyo --lat 35.68 --lon 139.69

# Generate the full combined report
python capstone_app.py report

# Help
python capstone_app.py --help
```

---

## 🧩 Architecture

```
capstone_app.py
│
├── @dataclass Stats              # container for descriptive statistics
├── @dataclass WeatherData        # parsed weather API response
│
├── DataAnalyser                  # loads dataset, computes all analytics
│   ├── compute_stats(field)      # NumPy-powered descriptive stats
│   ├── dept_breakdown()          # headcount, avg salary, total payroll per dept
│   ├── top_performers(n, by)     # ranked employee list
│   ├── correlation(a, b)         # Pearson correlation via np.corrcoef
│   └── print_table()             # formatted console table
│
├── WeatherClient                 # Open-Meteo REST API integration
│   └── fetch()                   # returns WeatherData or None (graceful fail)
│
├── ReportGenerator               # composes DataAnalyser + WeatherClient
│   └── generate()                # returns full report string
│
└── CLI (argparse)
    ├── stats   → cmd_stats()
    ├── weather → cmd_weather()
    └── report  → cmd_report()
```

---

## 💡 Design Principles Applied

| Principle | Implementation |
|-----------|---------------|
| **OOP** | `DataAnalyser`, `WeatherClient`, `ReportGenerator` classes |
| **Dataclasses** | `Stats` and `WeatherData` as typed data containers |
| **Type hints** | Throughout — `list[dict[str, Any]]`, `float \| None`, return types |
| **Structured logging** | `logging.getLogger` with timestamped, levelled output |
| **Error handling** | `try/except` at every I/O boundary — HTTP, JSON, network |
| **Modularity** | Each class is independently testable with no cross-dependencies |
| **NumPy integration** | All statistics computed via vectorised NumPy operations |

---

## 💡 Key Code Highlights

### Dataclass as Typed Data Container
```python
from dataclasses import dataclass, field

@dataclass
class Stats:
    field_name: str
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float

    def __str__(self) -> str:
        return f"  {self.field_name:<14} mean={self.mean:>10,.2f}  std={self.std:>9,.2f}"
```

### NumPy-Powered Analytics
```python
def compute_stats(self, field: str) -> Stats:
    values = np.array([r[field] for r in self._records], dtype=float)
    return Stats(
        field_name=field,
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=float(np.std(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
    )

def correlation(self, field_a: str, field_b: str) -> float:
    a = np.array([r[field_a] for r in self._records], dtype=float)
    b = np.array([r[field_b] for r in self._records], dtype=float)
    return float(np.corrcoef(a, b)[0, 1])
```

### Graceful API Client
```python
class WeatherClient:
    def fetch(self) -> WeatherData | None:
        try:
            response = requests.get(self._BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            return WeatherData(...)
        except (HTTPError, ConnectionError, Timeout) as exc:
            logger.error("Weather fetch failed: %s", exc)
            return None   # caller handles None — no crash
```

### argparse Subcommands
```python
parser = argparse.ArgumentParser(prog="capstone_app")
subparsers = parser.add_subparsers(dest="command", required=True)

subparsers.add_parser("stats",   help="Analyse the employee dataset.")
subparsers.add_parser("weather", help="Fetch live weather.")
subparsers.add_parser("report",  help="Generate full analytics report.")
```

---

## 📊 Sample Output

```
╔══════════════════════════════════════════════════════════╗
║   Al Aziz Technologies — Week 1 Capstone                 ║
║   CLI Analytics Tool  v1.0                               ║
╚══════════════════════════════════════════════════════════╝

── Descriptive Statistics ──
  salary         mean=   87,700.00  std=   11,024.97  min=   68,000.00  max=  105,000.00
  exp_years      mean=        5.00  std=        2.45  min=        1.00  max=        9.00
  score          mean=       84.90  std=        6.80  min=       73.00  max=       95.00

── Department Breakdown ──
  AI Research      headcount=4  avg_salary=PKR    97,250
  Data Science     headcount=3  avg_salary=PKR    83,000
  Engineering      headcount=3  avg_salary=PKR    72,667

── Top 3 Performers (by score) ──
  #1 Ghazal Butt            score=95
  #2 Jawad Mehmood          score=93
  #3 Ayesha Malik           score=91

── Correlations ──
  Salary ↔ Experience : 0.9823
  Salary ↔ Score      : 0.9714
```

---

## 🛠️ Skills Demonstrated

`Python OOP` · `Dataclasses` · `NumPy` · `REST API Integration` · `argparse CLI` · `Structured Logging` · `Statistical Analysis` · `Data Analytics` · `Production Error Handling` · `Type Hints` · `Modular Architecture`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
