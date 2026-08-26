"""
Week 1 | Day 4 — NumPy Basics
==============================
Internship at Al Aziz Technologies | AI Engineering Track

Topics covered:
    - Creating arrays (1-D, 2-D, 3-D) from lists, ranges, zeros/ones
    - Array shapes, ndim, dtype, itemsize
    - Indexing and slicing (including Boolean masking)
    - Broadcasting rules
    - Vectorised arithmetic — element-wise and matrix ops
    - Aggregation: sum, mean, std, min, max along axes
    - Linear algebra: dot product, matrix multiply, transpose, inverse
    - Practical AI use-case: manual feature normalisation

Run:
    python "Week 1/Day 4/numpy_basics.py"
"""

import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("day4.numpy")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ARRAY CREATION
# ══════════════════════════════════════════════════════════════════════════════
def demo_array_creation() -> None:
    logger.info("── Array Creation ──")
    print("\n── Array Creation ──")

    a1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    a2 = np.arange(0, 10, 2)                        # [0, 2, 4, 6, 8]
    a3 = np.linspace(0, 1, 6)                        # 6 evenly spaced
    a4 = np.zeros((3, 4))
    a5 = np.ones((2, 3), dtype=np.int32)
    a6 = np.eye(4)                                   # identity matrix
    a7 = np.full((2, 2), fill_value=7.0)
    rng = np.random.default_rng(seed=42)
    a8 = rng.random((3, 3))                          # uniform [0, 1)

    for name, arr in [("from list", a1), ("arange", a2), ("linspace", a3),
                      ("zeros(3,4)", a4), ("ones(2,3)", a5), ("eye(4)", a6),
                      ("full(2,2,7)", a7), ("random(3,3)", a8)]:
        print(f"  {name:<14} shape={str(arr.shape):<12} dtype={arr.dtype}")

    print(f"\n  a1 = {a1}")
    print(f"  a8 =\n{a8.round(3)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SHAPES, NDIM, DTYPE
# ══════════════════════════════════════════════════════════════════════════════
def demo_shape_info() -> None:
    logger.info("── Shape / ndim / dtype ──")
    print("\n── Shape Inspection ──")

    arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    print(f"  arr.shape    = {arr.shape}")
    print(f"  arr.ndim     = {arr.ndim}")
    print(f"  arr.dtype    = {arr.dtype}")
    print(f"  arr.size     = {arr.size}   (total elements)")
    print(f"  arr.itemsize = {arr.itemsize} bytes per element")
    print(f"  arr.nbytes   = {arr.nbytes} bytes total")

    flat = arr.flatten()
    col  = arr.reshape(-1, 1)  # column vector
    print(f"\n  Flatten → shape {flat.shape}")
    print(f"  Reshape(-1,1) → shape {col.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. INDEXING & SLICING
# ══════════════════════════════════════════════════════════════════════════════
def demo_indexing() -> None:
    logger.info("── Indexing & Slicing ──")
    print("\n── Indexing & Slicing ──")

    matrix = np.array([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
    ])
    print(f"  Matrix:\n{matrix}")
    print(f"\n  matrix[1, 2]    = {matrix[1, 2]}")
    print(f"  matrix[0, :]    = {matrix[0, :]}")
    print(f"  matrix[:, 2]    = {matrix[:, 2]}")
    print(f"  matrix[1:, 1:3] =\n{matrix[1:, 1:3]}")

    # Boolean masking
    print("\n  Boolean masking (elements > 60):")
    mask = matrix > 60
    print(f"  Mask:\n{mask}")
    print(f"  Selected: {matrix[mask]}")

    # Fancy indexing
    rows = np.array([0, 2])
    cols = np.array([1, 3])
    print(f"\n  Fancy index rows={rows}, cols={cols}: {matrix[rows, cols]}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BROADCASTING
# ══════════════════════════════════════════════════════════════════════════════
def demo_broadcasting() -> None:
    logger.info("── Broadcasting ──")
    print("\n── Broadcasting ──")

    # (3, 4) + (4,) — row vector broadcast
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    row_bias = np.array([1.0, 2.0, 3.0, 4.0])        # shape (4,)
    col_bias = np.array([[10.0], [20.0], [30.0]])     # shape (3, 1)

    print(f"  matrix shape        : {matrix.shape}")
    print(f"  matrix:\n{matrix}")
    print(f"\n  row_bias shape (4,) : {row_bias}")
    print(f"  matrix + row_bias:\n{matrix + row_bias}")
    print(f"\n  col_bias shape (3,1):\n{col_bias}")
    print(f"  matrix + col_bias:\n{matrix + col_bias}")

    # Outer product via broadcasting
    a = np.array([1, 2, 3]).reshape(-1, 1)   # (3, 1)
    b = np.array([10, 20, 30])               # (3,)
    outer = a * b                            # (3, 3)
    print(f"\n  Outer product (3,1)*(3,):\n{outer}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. VECTORISED OPERATIONS & AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
def demo_vectorised_ops() -> None:
    logger.info("── Vectorised Operations & Aggregation ──")
    print("\n── Vectorised Operations ──")

    rng = np.random.default_rng(seed=0)
    data = rng.integers(low=50, high=100, size=(5, 6)).astype(float)
    print(f"  data (5×6):\n{data}")

    # Element-wise ops
    print(f"\n  +10  (broadcast scalar):\n{data + 10}")
    print(f"  **2  (element-wise square):\n{data ** 2}")

    # Aggregation
    print(f"\n  Overall mean  : {data.mean():.2f}")
    print(f"  Column means  : {data.mean(axis=0).round(2)}")
    print(f"  Row sums      : {data.sum(axis=1)}")
    print(f"  Global std    : {data.std():.2f}")
    print(f"  Global min/max: {data.min():.0f} / {data.max():.0f}")

    # Universal functions (ufuncs)
    angles = np.linspace(0, np.pi, 5)
    print(f"\n  angles (rad): {angles.round(3)}")
    print(f"  sin          : {np.sin(angles).round(3)}")
    print(f"  exp          : {np.exp(np.array([0, 1, 2]))}")
    print(f"  log          : {np.log(np.array([1, np.e, np.e**2])).round(3)}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. LINEAR ALGEBRA
# ══════════════════════════════════════════════════════════════════════════════
def demo_linear_algebra() -> None:
    logger.info("── Linear Algebra ──")
    print("\n── Linear Algebra ──")

    A = np.array([[2, 1], [5, 3]], dtype=float)
    B = np.array([[1, 2], [3, 4]], dtype=float)
    v = np.array([1.0, 2.0])

    print(f"  A:\n{A}")
    print(f"\n  B:\n{B}")
    print(f"\n  A @ B (matrix multiply):\n{A @ B}")
    print(f"\n  A.T (transpose):\n{A.T}")
    print(f"\n  np.linalg.det(A)   = {np.linalg.det(A):.2f}")
    print(f"  np.linalg.inv(A)   =\n{np.linalg.inv(A)}")
    print(f"\n  dot(A, v)          = {A @ v}")

    # Eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"\n  Eigenvalues of A   : {eigenvalues.round(4)}")
    print(f"  Eigenvectors of A  :\n{eigenvectors.round(4)}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRACTICAL: FEATURE NORMALISATION (Min-Max Scaling)
# ══════════════════════════════════════════════════════════════════════════════
def demo_feature_normalisation() -> None:
    """
    Manually implement Min-Max and Z-Score normalisation using NumPy.
    This is exactly how Scikit-Learn's MinMaxScaler and StandardScaler work under the hood.
    """
    logger.info("── Feature Normalisation ──")
    print("\n── Practical: Feature Normalisation ──")

    # Simulated feature matrix: [age, salary(k), experience(yrs), score]
    X = np.array([
        [25,  45.0,  2, 72],
        [32,  85.0,  8, 88],
        [28,  62.0,  4, 65],
        [45, 120.0, 20, 91],
        [37,  95.0, 12, 79],
        [22,  38.0,  1, 60],
    ], dtype=float)

    feature_names = ["Age", "Salary(k)", "Experience", "Score"]
    print(f"  Raw X:\n  {'':>3} {' '.join(f'{n:>12}' for n in feature_names)}")
    for i, row in enumerate(X):
        print(f"  [{i}] {' '.join(f'{v:>12.1f}' for v in row)}")

    # Min-Max Scaling → [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_minmax = (X - X_min) / (X_max - X_min)

    print(f"\n  Min-Max Scaled X (range [0,1]):")
    for i, row in enumerate(X_minmax):
        print(f"  [{i}] {' '.join(f'{v:>12.4f}' for v in row)}")

    # Z-Score Standardisation → mean=0, std=1
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    X_std[X_std == 0] = 1          # avoid division by zero
    X_zscore = (X - X_mean) / X_std

    print(f"\n  Z-Score Standardised X (mean≈0, std≈1):")
    for i, row in enumerate(X_zscore):
        print(f"  [{i}] {' '.join(f'{v:>12.4f}' for v in row)}")

    print(f"\n  Verification — column means (should ≈ 0): {X_zscore.mean(axis=0).round(10)}")
    print(f"  Verification — column stds  (should ≈ 1): {X_zscore.std(axis=0).round(4)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "█" * 60)
    print("  WEEK 1 | DAY 4 — NumPy Basics")
    print("  Internship at Al Aziz Technologies")
    print("█" * 60)

    demo_array_creation()
    demo_shape_info()
    demo_indexing()
    demo_broadcasting()
    demo_vectorised_ops()
    demo_linear_algebra()
    demo_feature_normalisation()

    print("\n" + "=" * 60)
    logger.info("Day 4 — NumPy complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
