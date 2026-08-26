"""
Week 1 | Day 4 — File I/O, JSON, CSV & Logging
================================================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - Writing / reading plain text files
    - Reading and writing CSV with the csv module
    - Serialising / deserialising JSON
    - Structured logging (console + rotating file handler)
    - Exception handling for file operations
    - Environment variables with python-dotenv (explained via os.environ)

All data is generated inline — no external files required.

Run:
    python "Week 1/Day 4/file_io_json_csv.py"
"""

import csv
import json
import logging
import logging.handlers
import os
import sys
import tempfile
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = Path(tempfile.gettempdir()) / "alaziz_internship_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "day4.log"

_fmt = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=_fmt,
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=500_000, backupCount=2, encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("day4.file_io")


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATASETS
# ══════════════════════════════════════════════════════════════════════════════
EMPLOYEES: list[dict[str, Any]] = [
    {"id": 1, "name": "Ayesha Malik",  "dept": "AI Research",  "salary": 95000, "active": True},
    {"id": 2, "name": "Bilal Khan",    "dept": "Data Science", "salary": 82000, "active": True},
    {"id": 3, "name": "Chand Rehman",  "dept": "AI Research",  "salary": 91000, "active": False},
    {"id": 4, "name": "Dania Siddiqui","dept": "Engineering",  "salary": 78000, "active": True},
    {"id": 5, "name": "Ehsan Tariq",   "dept": "Data Science", "salary": 88000, "active": True},
]

EXPERIMENTS: list[dict[str, Any]] = [
    {"exp_id": "exp_001", "model": "LogisticRegression", "accuracy": 0.8721, "f1": 0.8534, "date": "2025-08-11"},
    {"exp_id": "exp_002", "model": "RandomForest",       "accuracy": 0.9214, "f1": 0.9187, "date": "2025-08-12"},
    {"exp_id": "exp_003", "model": "GradientBoosting",   "accuracy": 0.9352, "f1": 0.9301, "date": "2025-08-13"},
    {"exp_id": "exp_004", "model": "SVM",                "accuracy": 0.8903, "f1": 0.8841, "date": "2025-08-14"},
    {"exp_id": "exp_005", "model": "KNN",                "accuracy": 0.8511, "f1": 0.8479, "date": "2025-08-15"},
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. PLAIN TEXT FILE I/O
# ══════════════════════════════════════════════════════════════════════════════
def demo_text_file_io() -> None:
    """Write and read a plain text report."""
    logger.info("── Plain Text File I/O ──")

    tmp_path = Path(tempfile.gettempdir()) / "internship_report.txt"

    # Write
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write("=" * 50 + "\n")
            fh.write("Al Aziz Technologies — Internship Report\n")
            fh.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write("=" * 50 + "\n\n")
            for emp in EMPLOYEES:
                line = (
                    f"ID={emp['id']}  Name={emp['name']:<20}  "
                    f"Dept={emp['dept']:<15}  Salary=PKR {emp['salary']:,}\n"
                )
                fh.write(line)
        logger.info("Text file written → %s", tmp_path)
    except OSError as exc:
        logger.error("Failed to write text file: %s", exc)
        return

    # Read back
    print("\n── Text File Contents ──")
    try:
        with open(tmp_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        print(content)
    except OSError as exc:
        logger.error("Failed to read text file: %s", exc)

    # Append
    try:
        with open(tmp_path, "a", encoding="utf-8") as fh:
            fh.write(f"\nFooter: {len(EMPLOYEES)} records total.\n")
        logger.info("Appended footer to text file.")
    except OSError as exc:
        logger.error("Failed to append to text file: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CSV I/O
# ══════════════════════════════════════════════════════════════════════════════
def demo_csv_io() -> None:
    """Write a dataset to CSV (via StringIO) and read it back row by row."""
    logger.info("── CSV I/O ──")

    # ── Write to an in-memory CSV (StringIO avoids real file dependency) ──
    buffer = StringIO()
    fieldnames = ["id", "name", "dept", "salary", "active"]

    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(EMPLOYEES)

    csv_content = buffer.getvalue()
    logger.info("CSV written to StringIO buffer (%d chars).", len(csv_content))

    # ── Also write to a temp file ──
    tmp_csv = Path(tempfile.gettempdir()) / "employees.csv"
    try:
        with open(tmp_csv, "w", newline="", encoding="utf-8") as fh:
            fh.write(csv_content)
        logger.info("CSV saved → %s", tmp_csv)
    except OSError as exc:
        logger.error("CSV write failed: %s", exc)

    # ── Read back ──
    print("\n── CSV Read-back ──")
    buffer.seek(0)
    reader = csv.DictReader(buffer)
    rows: list[dict[str, str]] = list(reader)
    print(f"  {'ID':<4} {'Name':<22} {'Dept':<16} {'Salary':>8} {'Active':<6}")
    print(f"  {'-'*4} {'-'*22} {'-'*16} {'-'*8} {'-'*6}")
    for row in rows:
        print(
            f"  {row['id']:<4} {row['name']:<22} {row['dept']:<16} "
            f"{int(row['salary']):>8,} {row['active']:<6}"
        )
    logger.info("CSV read complete — %d rows.", len(rows))

    # ── Filter active employees ──
    active = [r for r in rows if r["active"].lower() == "true"]
    print(f"\n  Active employees ({len(active)}): {[r['name'] for r in active]}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. JSON I/O
# ══════════════════════════════════════════════════════════════════════════════
def demo_json_io() -> None:
    """Serialise experiment results to JSON and deserialise back."""
    logger.info("── JSON I/O ──")

    payload: dict[str, Any] = {
        "project": "Customer Churn Prediction",
        "intern": "Fatir Faraz",
        "generated_at": datetime.now().isoformat(),
        "experiments": EXPERIMENTS,
        "best_model": max(EXPERIMENTS, key=lambda e: e["accuracy"])["model"],
    }

    # ── Serialise ──
    json_str: str = json.dumps(payload, indent=2, ensure_ascii=False)
    logger.info("JSON serialised — %d chars.", len(json_str))

    # ── Save to temp file ──
    tmp_json = Path(tempfile.gettempdir()) / "experiments.json"
    try:
        with open(tmp_json, "w", encoding="utf-8") as fh:
            fh.write(json_str)
        logger.info("JSON saved → %s", tmp_json)
    except OSError as exc:
        logger.error("JSON write failed: %s", exc)
        return

    # ── Load & inspect ──
    print("\n── JSON Read-back ──")
    try:
        with open(tmp_json, "r", encoding="utf-8") as fh:
            loaded: dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("JSON read failed: %s", exc)
        return

    print(f"  Project     : {loaded['project']}")
    print(f"  Intern      : {loaded['intern']}")
    print(f"  Best model  : {loaded['best_model']}")
    print(f"  Experiments : {len(loaded['experiments'])}")
    print("\n  Experiment results:")
    for exp in loaded["experiments"]:
        print(
            f"    [{exp['exp_id']}] {exp['model']:<22} "
            f"acc={exp['accuracy']:.4f}  f1={exp['f1']:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENVIRONMENT VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
def demo_environment_variables() -> None:
    """Show how to read env vars safely (simulating .env usage)."""
    logger.info("── Environment Variables ──")
    print("\n── Environment Variables ──")

    # Simulate setting vars (in real use these come from .env via python-dotenv)
    os.environ.setdefault("APP_ENV", "development")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("MAX_RETRIES", "3")

    app_env: str = os.environ.get("APP_ENV", "production")
    log_level: str = os.environ.get("LOG_LEVEL", "WARNING")
    api_key: str = os.environ.get("API_KEY", "NOT_SET")   # would come from .env
    max_retries: int = int(os.environ.get("MAX_RETRIES", "3"))

    print(f"  APP_ENV     : {app_env}")
    print(f"  LOG_LEVEL   : {log_level}")
    print(f"  API_KEY     : {'*' * len(api_key) if api_key != 'NOT_SET' else api_key}")
    print(f"  MAX_RETRIES : {max_retries}")
    print("\n  Tip: use python-dotenv + a .env file so secrets never enter source code.")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOGGING LEVELS DEMO
# ══════════════════════════════════════════════════════════════════════════════
def demo_logging_levels() -> None:
    """Show all five standard logging levels."""
    demo_log = logging.getLogger("day4.levels")
    print("\n── Logging Levels ──")
    demo_log.debug   ("DEBUG   — fine-grained diagnostic info (dev only)")
    demo_log.info    ("INFO    — normal operational messages")
    demo_log.warning ("WARNING — something unexpected but non-fatal")
    demo_log.error   ("ERROR   — a serious problem; function may have failed")
    demo_log.critical("CRITICAL — program may not be able to continue")
    print(f"\n  Log file written to: {LOG_FILE}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█" * 60)
    print("  WEEK 1 | DAY 4 — File I/O, JSON, CSV & Logging")
    print("  Internship at Al Aziz Technologies")
    print("█" * 60)

    demo_text_file_io()
    demo_csv_io()
    demo_json_io()
    demo_environment_variables()
    demo_logging_levels()

    print("\n" + "=" * 60)
    logger.info("Day 4 — File I/O, JSON, CSV complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
