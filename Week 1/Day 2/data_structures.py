"""
Week 1 | Day 2 — Python Data Structures & Control Flow
========================================================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - Lists, Tuples, Sets, Dictionaries
    - Indexing, slicing, and mutability
    - if / elif / else
    - for loops, while loops, break, continue
    - List comprehensions and dictionary comprehensions
    - Nested data structures
    - Practical problems on structured data

Run:
    python "Week 1/Day 2/data_structures.py"
"""

import logging
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LISTS
# ═══════════════════════════════════════════════════════════════════════════════
def demo_lists() -> None:
    """Demonstrate list creation, indexing, slicing, and common methods."""
    logger.info("── Lists ──")

    # AI frameworks ordered by release year
    frameworks: list[str] = ["NumPy", "SciPy", "Pandas", "Scikit-Learn", "PyTorch", "TensorFlow", "LangChain"]

    print("\n── Lists ──")
    print(f"  Original   : {frameworks}")
    print(f"  First      : {frameworks[0]}")
    print(f"  Last       : {frameworks[-1]}")
    print(f"  Slice [2:5]: {frameworks[2:5]}")
    print(f"  Reversed   : {frameworks[::-1]}")
    print(f"  Length     : {len(frameworks)}")

    # Mutation
    frameworks.append("Transformers")
    frameworks.insert(0, "Matplotlib")
    popped = frameworks.pop()          # removes last
    frameworks.sort()

    print(f"  After sort : {frameworks}")
    print(f"  Popped     : {popped!r}")
    print(f"  Count 'NumPy': {frameworks.count('NumPy')}")
    print(f"  Index 'Pandas': {frameworks.index('Pandas')}")

    # List of numbers
    scores: list[int] = [88, 72, 95, 61, 84, 77, 90]
    print(f"\n  Scores : {scores}")
    print(f"  Max    : {max(scores)}")
    print(f"  Min    : {min(scores)}")
    print(f"  Sum    : {sum(scores)}")
    print(f"  Avg    : {sum(scores)/len(scores):.2f}")
    print(f"  Sorted : {sorted(scores, reverse=True)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TUPLES
# ═══════════════════════════════════════════════════════════════════════════════
def demo_tuples() -> None:
    """Show tuple immutability and common use-cases."""
    logger.info("── Tuples ──")

    # Tuples are immutable — great for fixed records
    employee: tuple[int, str, str, float] = (1001, "Fatir Faraz", "AI Engineer", 85_000.0)
    emp_id, name, role, salary = employee   # unpacking

    print("\n── Tuples ──")
    print(f"  Record   : {employee}")
    print(f"  Unpacked : id={emp_id}, name={name}, role={role}, salary={salary:,.0f}")
    print(f"  Immutable: cannot reassign elements (raises TypeError)")

    try:
        employee[0] = 9999  # type: ignore
    except TypeError as exc:
        print(f"  ✗ Caught expected error: {exc}")

    # Tuple of model configs (immutable hyperparams)
    configs: tuple[tuple[str, float, int], ...] = (
        ("LogisticRegression", 0.01, 1000),
        ("RandomForest", 0.001, 500),
        ("SVM", 0.1, 200),
    )
    print("\n  Model configs (name, lr, epochs):")
    for cfg in configs:
        print(f"    {cfg}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SETS
# ═══════════════════════════════════════════════════════════════════════════════
def demo_sets() -> None:
    """Demonstrate set operations: union, intersection, difference."""
    logger.info("── Sets ──")

    week1_topics: set[str] = {"Python", "NumPy", "Pandas", "APIs", "OOP"}
    week2_topics: set[str] = {"Pandas", "Scikit-Learn", "Matplotlib", "NumPy", "EDA"}

    print("\n── Sets ──")
    print(f"  Week 1 topics : {sorted(week1_topics)}")
    print(f"  Week 2 topics : {sorted(week2_topics)}")
    print(f"  Union (∪)     : {sorted(week1_topics | week2_topics)}")
    print(f"  Intersection  : {sorted(week1_topics & week2_topics)}")
    print(f"  Difference W1 : {sorted(week1_topics - week2_topics)}")
    print(f"  Symmetric diff: {sorted(week1_topics ^ week2_topics)}")

    # Sets deduplicate automatically
    raw_tags: list[str] = ["AI", "ML", "AI", "NLP", "ML", "CV", "NLP"]
    unique_tags: set[str] = set(raw_tags)
    print(f"\n  Raw tags  : {raw_tags}")
    print(f"  Unique    : {sorted(unique_tags)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DICTIONARIES
# ═══════════════════════════════════════════════════════════════════════════════
def demo_dictionaries() -> None:
    """Explore dictionaries as the backbone of structured AI data."""
    logger.info("── Dictionaries ──")

    # Intern profile
    intern: dict[str, Any] = {
        "name": "Fatir Faraz",
        "university": "Federal Urdu University, Islamabad",
        "field": "AI Engineering",
        "week": 1,
        "completed_days": [1, 2],
        "skills": {"Python", "n8n", "FastAPI"},
    }

    print("\n── Dictionaries ──")
    print(f"  Keys     : {list(intern.keys())}")
    print(f"  Values   : {[v for v in intern.values() if not isinstance(v, (set, list))]}")
    print(f"  Name     : {intern['name']}")
    print(f"  Get safe : {intern.get('mentor', 'TBD')}")

    # Update
    intern["week"] = 2
    intern["completed_days"].append(3)
    intern.setdefault("github", "Fatir002")

    print(f"\n  Updated intern record:")
    for key, value in intern.items():
        print(f"    {key:<20}: {value}")

    # Nested dict — model metrics store
    model_metrics: dict[str, dict[str, float]] = {
        "LogisticRegression": {"accuracy": 0.87, "f1": 0.85, "roc_auc": 0.91},
        "RandomForest":       {"accuracy": 0.92, "f1": 0.91, "roc_auc": 0.96},
        "SVM":                {"accuracy": 0.89, "f1": 0.88, "roc_auc": 0.93},
    }
    print("\n  Model Metrics:")
    print(f"  {'Model':<22} {'Accuracy':>9} {'F1':>6} {'ROC-AUC':>8}")
    print(f"  {'-'*22} {'-'*9} {'-'*6} {'-'*8}")
    for model, m in model_metrics.items():
        print(f"  {model:<22} {m['accuracy']:>9.3f} {m['f1']:>6.3f} {m['roc_auc']:>8.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONTROL FLOW
# ═══════════════════════════════════════════════════════════════════════════════
def demo_control_flow() -> None:
    """Show if/elif/else, for, while, break, and continue."""
    logger.info("── Control Flow ──")
    print("\n── if / elif / else ──")

    def classify_score(score: int) -> str:
        """Return a grade label for a numeric score."""
        if score >= 90:
            return "A — Excellent"
        elif score >= 80:
            return "B — Good"
        elif score >= 70:
            return "C — Satisfactory"
        elif score >= 60:
            return "D — Needs Improvement"
        else:
            return "F — Fail"

    test_scores: list[int] = [95, 83, 71, 64, 55]
    for s in test_scores:
        print(f"  Score {s}: {classify_score(s)}")

    # ── for loop ──
    print("\n── for loop (week progress) ──")
    days: list[str] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    tasks: list[str] = [
        "Python Fundamentals",
        "Data Structures",
        "Functions & OOP",
        "Files, APIs & NumPy",
        "Capstone Project",
    ]
    for day, task in zip(days, tasks):
        print(f"  {day:<12} → {task}")

    # ── while loop ──
    print("\n── while loop (training epochs) ──")
    epoch: int = 1
    loss: float = 1.0
    while loss > 0.1:
        loss *= 0.65
        print(f"  Epoch {epoch:>2}: loss = {loss:.4f}")
        epoch += 1
        if epoch > 20:
            logger.warning("Max epochs reached — breaking.")
            break
    print(f"  Converged after {epoch - 1} epochs.")

    # ── break / continue ──
    print("\n── break & continue (data filtering) ──")
    raw_data: list[int | None] = [10, None, 30, None, -5, 50, None, 70]
    clean: list[int] = []
    for item in raw_data:
        if item is None:
            logger.debug("Skipping None value.")
            continue          # skip missing values
        if item < 0:
            logger.warning("Negative value %d — stopping pipeline.", item)
            break             # negative sentinel → abort
        clean.append(item)
    print(f"  Raw   : {raw_data}")
    print(f"  Clean : {clean}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COMPREHENSIONS
# ═══════════════════════════════════════════════════════════════════════════════
def demo_comprehensions() -> None:
    """Demonstrate list and dictionary comprehensions."""
    logger.info("── Comprehensions ──")
    print("\n── List Comprehensions ──")

    numbers: list[int] = list(range(1, 11))

    squares: list[int] = [n ** 2 for n in numbers]
    evens: list[int] = [n for n in numbers if n % 2 == 0]
    normalised: list[float] = [round(n / max(numbers), 2) for n in numbers]

    print(f"  numbers    : {numbers}")
    print(f"  squares    : {squares}")
    print(f"  evens      : {evens}")
    print(f"  normalised : {normalised}")

    # Nested comprehension — multiplication table
    size: int = 4
    table: list[list[int]] = [[r * c for c in range(1, size + 1)] for r in range(1, size + 1)]
    print(f"\n  {size}×{size} Multiplication table:")
    for row in table:
        print("  " + "  ".join(f"{v:3}" for v in row))

    print("\n── Dictionary Comprehensions ──")
    words: list[str] = ["AI", "Machine", "Learning", "Deep", "Neural"]
    word_lengths: dict[str, int] = {w: len(w) for w in words}
    upper_map: dict[str, str] = {w: w.upper() for w in words if len(w) > 3}

    print(f"  Word lengths : {word_lengths}")
    print(f"  Long words   : {upper_map}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. NESTED STRUCTURES (AI-flavoured dataset)
# ═══════════════════════════════════════════════════════════════════════════════
def demo_nested_structures() -> None:
    """Work with a nested list-of-dicts dataset — common in real AI pipelines."""
    logger.info("── Nested Data Structures ──")

    # Simulated dataset: employee records
    employees: list[dict[str, Any]] = [
        {"id": 1, "name": "Ayesha",  "dept": "AI Research",  "salary": 95_000, "skills": ["Python", "PyTorch"]},
        {"id": 2, "name": "Bilal",   "dept": "Data Science", "salary": 82_000, "skills": ["Pandas", "Scikit-Learn"]},
        {"id": 3, "name": "Chand",   "dept": "AI Research",  "salary": 91_000, "skills": ["Python", "TensorFlow"]},
        {"id": 4, "name": "Dania",   "dept": "Engineering",  "salary": 78_000, "skills": ["FastAPI", "Docker"]},
        {"id": 5, "name": "Ehsan",   "dept": "Data Science", "salary": 88_000, "skills": ["R", "SQL", "Pandas"]},
    ]

    print("\n── Nested Dataset: Employee Records ──")
    # Filter AI Research employees
    ai_team: list[dict[str, Any]] = [e for e in employees if e["dept"] == "AI Research"]
    print(f"  AI Research team: {[e['name'] for e in ai_team]}")

    # Average salary
    avg_salary: float = sum(e["salary"] for e in employees) / len(employees)
    print(f"  Average salary  : PKR {avg_salary:,.0f}")

    # Sort by salary descending
    ranked = sorted(employees, key=lambda e: e["salary"], reverse=True)
    print("\n  Salary ranking:")
    for rank, emp in enumerate(ranked, 1):
        print(f"    {rank}. {emp['name']:<10} ({emp['dept']:<15}) PKR {emp['salary']:,}")

    # All unique skills across team
    all_skills: set[str] = {skill for emp in employees for skill in emp["skills"]}
    print(f"\n  Unique skills across team: {sorted(all_skills)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Run all Day 2 demonstrations."""
    print("\n" + "█" * 60)
    print("  WEEK 1 | DAY 2 — Data Structures & Control Flow")
    print("  Internship at Al Aziz Technologies")
    print("█" * 60)

    demo_lists()
    demo_tuples()
    demo_sets()
    demo_dictionaries()
    demo_control_flow()
    demo_comprehensions()
    demo_nested_structures()

    print("\n" + "=" * 60)
    logger.info("Day 2 complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unexpected error: %s", exc, exc_info=True)
        sys.exit(1)
