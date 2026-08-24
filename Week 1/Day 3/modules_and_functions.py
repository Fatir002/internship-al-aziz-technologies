"""
Week 1 | Day 3 — Functions, Modules & Scope
============================================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - Defining functions with parameters, defaults, *args, **kwargs
    - Return values and multiple returns
    - Lambda functions
    - Variable scope (LEGB rule)
    - Importing standard-library and custom modules
    - Virtual environments & pip (explained in comments)

Run:
    python "Week 1/Day 3/modules_and_functions.py"
"""

import logging
import math
import sys
from functools import reduce
from typing import Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def greet(name: str, role: str = "AI Engineer") -> str:
    """Return a greeting string.

    Args:
        name: Person's name.
        role: Job role (default 'AI Engineer').

    Returns:
        Formatted greeting.
    """
    return f"Hello, {name}! Welcome to the {role} internship."


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32


def divide(a: float, b: float) -> float | None:
    """Safe division — returns None on zero divisor."""
    if b == 0:
        logger.warning("Division by zero attempted.")
        return None
    return a / b


def minmax(numbers: list[float]) -> tuple[float, float]:
    """Return (min, max) of a list — demonstrates multiple return values."""
    if not numbers:
        raise ValueError("Cannot find minmax of an empty list.")
    return min(numbers), max(numbers)


def demo_basic_functions() -> None:
    logger.info("── Basic Functions ──")
    print("\n── Basic Functions ──")
    print(f"  {greet('Fatir')}")
    print(f"  {greet('Ayesha', role='Data Scientist')}")
    print(f"  100°C = {celsius_to_fahrenheit(100):.1f}°F")
    print(f"  10 / 3 = {divide(10, 3):.4f}")
    print(f"  10 / 0 = {divide(10, 0)}")
    lo, hi = minmax([4.2, 1.1, 9.8, 3.3, 7.5])
    print(f"  minmax([4.2,1.1,9.8,3.3,7.5]) → min={lo}, max={hi}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. *args AND **kwargs
# ═══════════════════════════════════════════════════════════════════════════════
def summarise_scores(*scores: float) -> dict[str, float]:
    """Accept any number of scores and return summary statistics.

    Args:
        *scores: Variable positional arguments (floats).

    Returns:
        Dict with count, mean, min, max, std.
    """
    n = len(scores)
    if n == 0:
        return {}
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    return {
        "count": float(n),
        "mean": round(mean, 4),
        "min": min(scores),
        "max": max(scores),
        "std": round(math.sqrt(variance), 4),
    }


def build_model_config(model_name: str, **hyperparams: Any) -> dict[str, Any]:
    """Build a model configuration dict from keyword arguments.

    Args:
        model_name: Name of the ML model.
        **hyperparams: Arbitrary hyperparameter key-value pairs.

    Returns:
        Config dictionary.
    """
    return {"model": model_name, "params": hyperparams}


def demo_args_kwargs() -> None:
    logger.info("── *args / **kwargs ──")
    print("\n── *args / **kwargs ──")

    stats = summarise_scores(88, 72, 95, 61, 84, 77, 90)
    print("  Score summary:")
    for k, v in stats.items():
        print(f"    {k:<8}: {v}")

    cfg = build_model_config(
        "RandomForest",
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        random_state=42,
    )
    print(f"\n  Model config: {cfg}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LAMBDA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def demo_lambdas() -> None:
    logger.info("── Lambda Functions ──")
    print("\n── Lambda Functions ──")

    square: Callable[[float], float] = lambda x: x ** 2
    normalize: Callable[[float, float, float], float] = (
        lambda x, lo, hi: (x - lo) / (hi - lo) if hi != lo else 0.0
    )

    numbers = [3, 7, 1, 9, 4, 6, 2, 8, 5]
    print(f"  Numbers     : {numbers}")
    print(f"  Squared     : {list(map(square, numbers))}")
    print(f"  Evens only  : {list(filter(lambda n: n % 2 == 0, numbers))}")
    print(f"  Product     : {reduce(lambda a, b: a * b, numbers)}")

    models = [
        {"name": "LR",  "accuracy": 0.87},
        {"name": "RF",  "accuracy": 0.92},
        {"name": "SVM", "accuracy": 0.89},
    ]
    best = max(models, key=lambda m: m["accuracy"])
    print(f"\n  Best model  : {best['name']} (accuracy={best['accuracy']})")

    raw_scores = [61.0, 84.0, 95.0, 72.0, 77.0]
    lo_s, hi_s = min(raw_scores), max(raw_scores)
    normalised = [round(normalize(s, lo_s, hi_s), 3) for s in raw_scores]
    print(f"  Raw scores  : {raw_scores}")
    print(f"  Normalised  : {normalised}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SCOPE — LEGB RULE
# ═══════════════════════════════════════════════════════════════════════════════
GLOBAL_THRESHOLD: float = 0.5   # Module-level (Global) scope


def demo_scope() -> None:
    """Demonstrate Local → Enclosing → Global → Built-in scope resolution."""
    logger.info("── Scope (LEGB) ──")
    print("\n── Scope (LEGB Rule) ──")

    # Local
    local_var: str = "I am local to demo_scope()"
    print(f"  Local    : {local_var}")

    # Global read
    print(f"  Global   : GLOBAL_THRESHOLD = {GLOBAL_THRESHOLD}")

    # Closure — enclosing scope
    def make_threshold_checker(threshold: float) -> Callable[[float], bool]:
        """Return a closure that checks whether a value exceeds the threshold."""
        def check(value: float) -> bool:
            return value > threshold   # 'threshold' is in enclosing scope
        return check

    above_half = make_threshold_checker(0.5)
    test_probs = [0.3, 0.55, 0.48, 0.72, 0.91]
    results = [(p, above_half(p)) for p in test_probs]
    print(f"  Closure threshold=0.5 checks:")
    for prob, passed in results:
        symbol = "✓" if passed else "✗"
        print(f"    {symbol} p={prob} → {'above' if passed else 'below'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STANDARD LIBRARY MODULES
# ═══════════════════════════════════════════════════════════════════════════════
def demo_stdlib_modules() -> None:
    """Show practical use of math, random, datetime, and os modules."""
    import os
    import random
    from datetime import date, datetime

    logger.info("── Standard Library Modules ──")
    print("\n── Standard Library Modules ──")

    # math
    print(f"  math.pi     = {math.pi:.6f}")
    print(f"  math.e      = {math.e:.6f}")
    print(f"  math.log(100, 10) = {math.log(100, 10)}")
    print(f"  math.sqrt(144)    = {math.sqrt(144)}")
    print(f"  math.ceil(3.2)    = {math.ceil(3.2)}")
    print(f"  math.floor(3.9)   = {math.floor(3.9)}")

    # random (seeded for reproducibility)
    random.seed(42)
    print(f"\n  random.randint(1,100) = {random.randint(1, 100)}")
    population = list(range(1, 11))
    sample = random.sample(population, 5)
    print(f"  random.sample(1-10, 5) = {sample}")
    random.shuffle(population)
    print(f"  random.shuffle(1-10)   = {population}")

    # datetime
    today = date.today()
    now = datetime.now()
    print(f"\n  date.today()   = {today}")
    print(f"  datetime.now() = {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # os
    cwd = os.getcwd()
    print(f"\n  os.getcwd()    = {cwd}")
    print(f"  os.sep         = {os.sep!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HIGHER-ORDER FUNCTIONS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def pipeline_demo() -> None:
    """Chain map/filter/reduce to process a dataset — common in AI preprocessing."""
    logger.info("── Functional Pipeline ──")
    print("\n── Functional Pipeline (map → filter → reduce) ──")

    raw: list[str] = ["88", "NaN", "72", "95", "N/A", "61", "84", "77", "90"]
    print(f"  Raw strings : {raw}")

    # Step 1: parse (map)
    def safe_parse(s: str) -> float | None:
        try:
            return float(s)
        except ValueError:
            return None

    parsed: list[float | None] = list(map(safe_parse, raw))
    print(f"  Parsed      : {parsed}")

    # Step 2: drop NaN (filter)
    valid: list[float] = [v for v in parsed if v is not None]
    print(f"  Valid       : {valid}")

    # Step 3: aggregate (reduce)
    total: float = reduce(lambda a, b: a + b, valid)
    mean: float = total / len(valid)
    print(f"  Total       : {total}")
    print(f"  Mean        : {mean:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█" * 60)
    print("  WEEK 1 | DAY 3 — Functions, Modules & Scope")
    print("  Internship at Al Aziz Technologies")
    print("█" * 60)

    demo_basic_functions()
    demo_args_kwargs()
    demo_lambdas()
    demo_scope()
    demo_stdlib_modules()
    pipeline_demo()

    print("\n" + "=" * 60)
    logger.info("Day 3 — Functions & Modules complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unexpected error: %s", exc, exc_info=True)
        sys.exit(1)
