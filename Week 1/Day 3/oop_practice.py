"""
Week 1 | Day 3 — Object-Oriented Programming Practice
======================================================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - Classes, __init__, instance vs class attributes
    - Instance methods, class methods, static methods
    - Encapsulation with _ / __ prefixes
    - Inheritance and method overriding
    - __str__, __repr__, __len__ dunder methods
    - Practical OOP: a mini ML-experiment tracker

Run:
    python "Week 1/Day 3/oop_practice.py"
"""

import logging
import sys
from datetime import datetime
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASE CLASS — DataRecord
# ═══════════════════════════════════════════════════════════════════════════════
class DataRecord:
    """Base class representing a generic labelled data record.

    Attributes:
        record_id: Unique identifier.
        created_at: Timestamp of creation.
    """

    _total_records: int = 0   # class attribute — shared across all instances

    def __init__(self, record_id: int) -> None:
        self.record_id: int = record_id
        self.created_at: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        DataRecord._total_records += 1

    # ── dunder methods ──
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.record_id})"

    def __str__(self) -> str:
        return f"[Record #{self.record_id} | created {self.created_at}]"

    # ── class method ──
    @classmethod
    def total_created(cls) -> int:
        """Return the total number of DataRecord instances ever created."""
        return cls._total_records

    # ── static method ──
    @staticmethod
    def validate_id(record_id: Any) -> bool:
        """Return True if record_id is a positive integer."""
        return isinstance(record_id, int) and record_id > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EMPLOYEE — inherits DataRecord
# ═══════════════════════════════════════════════════════════════════════════════
class Employee(DataRecord):
    """An employee record with name, department, and salary.

    Inherits:
        DataRecord
    """

    def __init__(
        self,
        record_id: int,
        name: str,
        department: str,
        salary: float,
    ) -> None:
        super().__init__(record_id)
        self.name: str = name
        self.department: str = department
        self._salary: float = salary        # 'protected' by convention

    # ── property — controlled access to salary ──
    @property
    def salary(self) -> float:
        """Salary in PKR."""
        return self._salary

    @salary.setter
    def salary(self, amount: float) -> None:
        if amount < 0:
            raise ValueError(f"Salary cannot be negative: {amount}")
        logger.info("Salary updated for %s: PKR %.0f → PKR %.0f", self.name, self._salary, amount)
        self._salary = amount

    def annual_salary(self) -> float:
        """Return the annualised salary."""
        return self._salary * 12

    def give_raise(self, percent: float) -> None:
        """Apply a percentage raise."""
        if not 0 < percent <= 100:
            raise ValueError(f"Raise percent must be between 0 and 100, got {percent}.")
        increase = self._salary * (percent / 100)
        self._salary += increase
        logger.info("%s received a %.1f%% raise → new salary PKR %.0f", self.name, percent, self._salary)

    def __str__(self) -> str:
        return (
            f"Employee(id={self.record_id}, name={self.name!r}, "
            f"dept={self.department!r}, salary=PKR {self._salary:,.0f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ML_EXPERIMENT — inherits DataRecord
# ═══════════════════════════════════════════════════════════════════════════════
class MLExperiment(DataRecord):
    """Track a single machine-learning experiment run.

    Attributes:
        model_name: Algorithm used.
        hyperparams: Dict of hyperparameter values.
        metrics: Dict of evaluation metric values.
    """

    def __init__(self, record_id: int, model_name: str, hyperparams: dict[str, Any]) -> None:
        super().__init__(record_id)
        self.model_name: str = model_name
        self.hyperparams: dict[str, Any] = hyperparams
        self.metrics: dict[str, float] = {}
        self._status: str = "pending"

    def log_metrics(self, **metrics: float) -> None:
        """Record evaluation metrics for this experiment."""
        self.metrics.update(metrics)
        self._status = "complete"
        logger.info("Experiment #%d (%s) metrics logged: %s", self.record_id, self.model_name, metrics)

    def is_complete(self) -> bool:
        return self._status == "complete"

    def summary(self) -> str:
        """Return a formatted summary string."""
        metric_str = "  |  ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (
            f"Exp #{self.record_id:>3} | {self.model_name:<22} | "
            f"Status: {self._status:<9} | {metric_str}"
        )

    def __str__(self) -> str:
        return f"MLExperiment(#{self.record_id}, model={self.model_name!r}, status={self._status!r})"

    def __repr__(self) -> str:
        return f"MLExperiment(id={self.record_id}, model={self.model_name!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXPERIMENT_TRACKER — composes MLExperiment objects
# ═══════════════════════════════════════════════════════════════════════════════
class ExperimentTracker:
    """A container that manages a collection of ML experiments.

    Demonstrates composition and __len__, __iter__ dunder methods.
    """

    def __init__(self, project_name: str) -> None:
        self.project_name: str = project_name
        self._experiments: list[MLExperiment] = []

    def add(self, experiment: MLExperiment) -> None:
        """Add an experiment to the tracker."""
        self._experiments.append(experiment)

    def best(self, metric: str = "accuracy") -> MLExperiment | None:
        """Return the experiment with the highest value for a given metric."""
        complete = [e for e in self._experiments if e.is_complete() and metric in e.metrics]
        if not complete:
            return None
        return max(complete, key=lambda e: e.metrics[metric])

    def __len__(self) -> int:
        return len(self._experiments)

    def __iter__(self):
        return iter(self._experiments)

    def __str__(self) -> str:
        return f"ExperimentTracker(project={self.project_name!r}, runs={len(self)})"

    def print_leaderboard(self, metric: str = "accuracy") -> None:
        """Print a ranked table of completed experiments."""
        complete = sorted(
            [e for e in self._experiments if e.is_complete()],
            key=lambda e: e.metrics.get(metric, 0.0),
            reverse=True,
        )
        print(f"\n  ── Leaderboard: {self.project_name} (ranked by {metric}) ──")
        if not complete:
            print("  No completed experiments yet.")
            return
        for rank, exp in enumerate(complete, 1):
            score = exp.metrics.get(metric, float("nan"))
            print(f"  #{rank}  {exp.summary()}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMOS
# ═══════════════════════════════════════════════════════════════════════════════
def demo_employee() -> None:
    logger.info("── Employee OOP Demo ──")
    print("\n── Employee Class ──")

    emp1 = Employee(101, "Fatir Faraz",  "AI Research", 75_000)
    emp2 = Employee(102, "Ayesha Malik", "Data Science", 85_000)
    emp3 = Employee(103, "Bilal Khan",   "Engineering",  70_000)

    for emp in [emp1, emp2, emp3]:
        print(f"  {emp}")

    print(f"\n  Total records created: {DataRecord.total_created()}")
    print(f"  ID valid (101): {DataRecord.validate_id(101)}")
    print(f"  ID valid (-1) : {DataRecord.validate_id(-1)}")

    print(f"\n  {emp1.name} annual salary: PKR {emp1.annual_salary():,.0f}")
    emp1.give_raise(10)
    print(f"  After 10% raise: PKR {emp1.salary:,.0f}")

    try:
        emp1.salary = -5000
    except ValueError as exc:
        print(f"  ✗ Caught: {exc}")


def demo_experiment_tracker() -> None:
    logger.info("── Experiment Tracker OOP Demo ──")
    print("\n── ExperimentTracker Class ──")

    tracker = ExperimentTracker("Customer Churn Prediction")

    # Define experiments
    experiments_config: list[tuple[str, dict[str, Any]]] = [
        ("LogisticRegression", {"C": 1.0, "max_iter": 200}),
        ("RandomForestClassifier", {"n_estimators": 100, "max_depth": 8, "random_state": 42}),
        ("GradientBoosting", {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 4}),
        ("SVM", {"C": 10.0, "kernel": "rbf"}),
        ("KNeighborsClassifier", {"n_neighbors": 7}),
    ]

    # Simulate logged metrics (would come from actual training in later weeks)
    simulated_metrics: list[dict[str, float]] = [
        {"accuracy": 0.8721, "f1": 0.8534, "roc_auc": 0.9102},
        {"accuracy": 0.9214, "f1": 0.9187, "roc_auc": 0.9631},
        {"accuracy": 0.9352, "f1": 0.9301, "roc_auc": 0.9712},
        {"accuracy": 0.8903, "f1": 0.8841, "roc_auc": 0.9388},
        {"accuracy": 0.8511, "f1": 0.8479, "roc_auc": 0.9041},
    ]

    for i, ((name, params), metrics) in enumerate(zip(experiments_config, simulated_metrics), 1):
        exp = MLExperiment(i, name, params)
        exp.log_metrics(**metrics)
        tracker.add(exp)

    print(f"\n  Tracker : {tracker}")
    print(f"  Total runs: {len(tracker)}")

    tracker.print_leaderboard(metric="accuracy")

    winner = tracker.best(metric="roc_auc")
    if winner:
        print(f"\n  Best by ROC-AUC: {winner.model_name} ({winner.metrics['roc_auc']:.4f})")


def demo_inheritance_chain() -> None:
    """Show repr and str of the full inheritance chain."""
    logger.info("── Inheritance & Dunder Methods ──")
    print("\n── Inheritance Chain ──")
    emp = Employee(201, "Demo User", "R&D", 60_000)
    exp = MLExperiment(1, "NaiveBayes", {"var_smoothing": 1e-9})
    print(f"  repr(emp) : {repr(emp)}")
    print(f"  str(emp)  : {str(emp)}")
    print(f"  repr(exp) : {repr(exp)}")
    print(f"  str(exp)  : {str(exp)}")
    print(f"  isinstance(emp, DataRecord): {isinstance(emp, DataRecord)}")
    print(f"  isinstance(exp, DataRecord): {isinstance(exp, DataRecord)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█" * 60)
    print("  WEEK 1 | DAY 3 — Object-Oriented Programming")
    print("  Internship at Al Aziz Technologies")
    print("█" * 60)

    demo_inheritance_chain()
    demo_employee()
    demo_experiment_tracker()

    print("\n" + "=" * 60)
    logger.info("Day 3 — OOP complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unexpected error: %s", exc, exc_info=True)
        sys.exit(1)
