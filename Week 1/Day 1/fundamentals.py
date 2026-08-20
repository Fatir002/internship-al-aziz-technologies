"""
Week 1 | Day 1 — Python Fundamentals
=====================================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - What Python is and why it dominates AI development
    - Variables and dynamic typing
    - Core data types: int, float, str, bool, NoneType
    - Type conversion (implicit and explicit)
    - Arithmetic, comparison, logical, and bitwise operators
    - String operations and f-strings
    - Input / output  (output demonstrated; input mocked for non-interactive runs)
    - Comments and docstrings
    - Basic try/except error handling

Run:
    python "Week 1/Day 1/fundamentals.py"
"""

import logging
import sys

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WHY PYTHON FOR AI?
# ═══════════════════════════════════════════════════════════════════════════════
def why_python_for_ai() -> None:
    """Print a brief explanation of Python's role in AI."""
    reasons: list[str] = [
        "Readable syntax — easy to prototype ideas quickly",
        "Rich ecosystem: NumPy, Pandas, PyTorch, Scikit-Learn, LangChain …",
        "Massive community and open-source model libraries",
        "First-class support from every major AI/ML cloud provider",
        "Glue language: connects C/C++ backends with high-level logic",
    ]
    print("\n" + "=" * 60)
    print("  WHY PYTHON IS WIDELY USED IN AI")
    print("=" * 60)
    for i, reason in enumerate(reasons, 1):
        print(f"  {i}. {reason}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VARIABLES AND DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════
def demo_variables_and_types() -> None:
    """Demonstrate Python's core built-in data types."""
    logger.info("── Variables & Data Types ──")

    # Integer
    intern_id: int = 42
    # Float
    learning_rate: float = 0.001
    # String
    intern_name: str = "Fatir Faraz"
    # Boolean
    is_enrolled: bool = True
    # NoneType
    mentor: None = None

    print("\n── Core Data Types ──")
    variables = {
        "intern_id": intern_id,
        "learning_rate": learning_rate,
        "intern_name": intern_name,
        "is_enrolled": is_enrolled,
        "mentor": mentor,
    }
    for name, value in variables.items():
        print(f"  {name:<16} = {value!r:<20}  type={type(value).__name__}")

    # Python is dynamically typed — variables can be rebound
    x = 100
    logger.info("x is %d (int)", x)
    x = "now a string"  # legal in Python
    logger.info("x is %r (str) — dynamic typing in action", x)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TYPE CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════
def demo_type_conversion() -> None:
    """Show explicit (casting) and implicit type conversion."""
    logger.info("── Type Conversion ──")

    # Explicit conversion
    pi_str: str = "3.14159"
    pi_float: float = float(pi_str)       # str → float
    pi_int: int = int(pi_float)           # float → int  (truncates, not rounds)
    back_to_str: str = str(pi_int)        # int → str

    print("\n── Type Conversion ──")
    print(f"  Original string : {pi_str!r}")
    print(f"  → float         : {pi_float}")
    print(f"  → int (trunc)   : {pi_int}")
    print(f"  → str again     : {back_to_str!r}")

    # Boolean conversion
    print(f"\n  bool(0)={bool(0)}  bool(1)={bool(1)}  bool('')={bool('')}  bool('hi')={bool('hi')}")
    print(f"  int(True)={int(True)}   int(False)={int(False)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════
def demo_operators() -> None:
    """Demonstrate arithmetic, comparison, logical, and bitwise operators."""
    logger.info("── Operators ──")
    a, b = 17, 5

    print("\n── Arithmetic Operators ──")
    print(f"  {a} + {b}  = {a + b}")
    print(f"  {a} - {b}  = {a - b}")
    print(f"  {a} * {b}  = {a * b}")
    print(f"  {a} / {b}  = {a / b}   (true division → float)")
    print(f"  {a} // {b} = {a // b}   (floor division)")
    print(f"  {a} % {b}  = {a % b}    (modulo)")
    print(f"  {a} ** {b} = {a ** b}  (exponentiation)")

    print("\n── Comparison Operators ──")
    for op, result in [
        (f"{a} == {b}", a == b),
        (f"{a} != {b}", a != b),
        (f"{a} > {b}", a > b),
        (f"{a} < {b}", a < b),
        (f"{a} >= {b}", a >= b),
        (f"{a} <= {b}", a <= b),
    ]:
        print(f"  {op:<10} → {result}")

    print("\n── Logical Operators ──")
    t, f = True, False
    print(f"  True and False = {t and f}")
    print(f"  True or  False = {t or f}")
    print(f"  not True       = {not t}")

    print("\n── Bitwise Operators ──")
    x, y = 0b1010, 0b1100   # 10 and 12 in binary
    print(f"  {x:04b} & {y:04b} = {x & y:04b}  ({x} & {y}  = {x & y})")
    print(f"  {x:04b} | {y:04b} = {x | y:04b}  ({x} | {y}  = {x | y})")
    print(f"  {x:04b} ^ {y:04b} = {x ^ y:04b}  ({x} ^ {y}  = {x ^ y})")
    print(f"  {x:04b} << 1  = {x << 1:04b}  ({x} << 1 = {x << 1})")
    print(f"  {x:04b} >> 1  = {x >> 1:04b}  ({x} >> 1 = {x >> 1})")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STRINGS AND F-STRINGS
# ═══════════════════════════════════════════════════════════════════════════════
def demo_strings() -> None:
    """Explore Python string operations and f-string formatting."""
    logger.info("── Strings ──")

    name: str = "Al Aziz Technologies"
    greeting: str = f"Welcome to {name}!"

    print("\n── String Operations ──")
    print(f"  Original  : {name!r}")
    print(f"  Upper     : {name.upper()!r}")
    print(f"  Lower     : {name.lower()!r}")
    print(f"  Length    : {len(name)}")
    print(f"  Replace   : {name.replace('Al Aziz', 'AAT')!r}")
    print(f"  Split     : {name.split()}")
    print(f"  Startswith: {name.startswith('Al')}")
    print(f"  Contains  : {'Aziz' in name}")
    print(f"  Slice [3:8]: {name[3:8]!r}")
    print(f"  Reversed  : {name[::-1]!r}")
    print(f"\n  F-string  : {greeting}")

    # Multi-line string
    bio: str = (
        "Fatir Faraz\n"
        "AI Engineering Intern\n"
        "Al Aziz Technologies"
    )
    print(f"\n  Multi-line:\n{bio}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INPUT / OUTPUT  (output only in this script — no blocking stdin call)
# ═══════════════════════════════════════════════════════════════════════════════
def demo_input_output() -> None:
    """Demonstrate Python print formatting and simulated input."""
    logger.info("── Input / Output ──")

    # Simulated user data (avoids blocking input() in automated runs)
    simulated_name: str = "Fatir"
    simulated_age: int = 22

    print("\n── Output Formatting ──")
    print(f"  Name : {simulated_name}")
    print(f"  Age  : {simulated_age}")
    print(f"  {'Item':<15} {'Price':>8} {'In Stock':^10}")
    print(f"  {'-'*15} {'-'*8} {'-'*10}")
    print(f"  {'NumPy':<15} {'Free':>8} {'✓':^10}")
    print(f"  {'PyTorch':<15} {'Free':>8} {'✓':^10}")
    print(f"  {'GPT-4o API':<15} {'$0.005/k':>8} {'✓':^10}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BASIC ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════
def demo_error_handling() -> None:
    """Introduce try/except/else/finally error handling."""
    logger.info("── Error Handling ──")
    print("\n── Basic try / except / else / finally ──")

    test_cases: list[tuple[str, str]] = [
        ("42", "valid integer"),
        ("3.14", "float string — ValueError"),
        ("hello", "non-numeric — ValueError"),
    ]

    for raw, description in test_cases:
        try:
            result: int = int(raw)
        except ValueError as exc:
            logger.warning("Could not convert %r → int: %s", raw, exc)
            print(f"  ✗ {description}: could not convert {raw!r}")
        else:
            print(f"  ✓ {description}: parsed value = {result}")
        finally:
            pass  # cleanup would go here (e.g. close file handles)

    # ZeroDivisionError
    try:
        _ = 10 / 0
    except ZeroDivisionError:
        logger.error("Division by zero caught — handled gracefully.")
        print("  ✗ ZeroDivisionError: caught and handled.")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EXERCISES
# ═══════════════════════════════════════════════════════════════════════════════
def exercises() -> None:
    """
    Solve three beginner exercises to reinforce Day 1 concepts.

    Exercise 1: Temperature converter (Celsius → Fahrenheit)
    Exercise 2: Simple interest calculator
    Exercise 3: FizzBuzz (1–20)
    """
    print("\n" + "=" * 60)
    print("  EXERCISES")
    print("=" * 60)

    # Exercise 1 — Temperature Converter
    print("\n[Exercise 1] Celsius → Fahrenheit")
    for celsius in [0, 20, 37, 100]:
        fahrenheit: float = (celsius * 9 / 5) + 32
        print(f"  {celsius:>4}°C = {fahrenheit:.1f}°F")

    # Exercise 2 — Simple Interest
    print("\n[Exercise 2] Simple Interest  (P=10000, R=8%, T=3 years)")
    principal: float = 10_000.0
    rate: float = 8.0
    time: float = 3.0
    interest: float = (principal * rate * time) / 100
    total: float = principal + interest
    print(f"  Interest = PKR {interest:,.2f}")
    print(f"  Total    = PKR {total:,.2f}")

    # Exercise 3 — FizzBuzz
    print("\n[Exercise 3] FizzBuzz (1–20)")
    output: list[str] = []
    for n in range(1, 21):
        if n % 15 == 0:
            output.append("FizzBuzz")
        elif n % 3 == 0:
            output.append("Fizz")
        elif n % 5 == 0:
            output.append("Buzz")
        else:
            output.append(str(n))
    print("  " + ", ".join(output))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Entry point — run all Day 1 demonstrations in sequence."""
    print("\n" + "█" * 60)
    print("  WEEK 1 | DAY 1 — Python Fundamentals")
    print("  Internship at Al Aziz Technologies")
    print("█" * 60)

    why_python_for_ai()
    demo_variables_and_types()
    demo_type_conversion()
    demo_operators()
    demo_strings()
    demo_input_output()
    demo_error_handling()
    exercises()

    print("\n" + "=" * 60)
    logger.info("Day 1 complete. All demonstrations ran successfully.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unexpected error: %s", exc, exc_info=True)
        sys.exit(1)
