# 🔢 Week 1 | Day 4 — NumPy Fundamentals for AI/ML

> **Internship at Al Aziz Technologies · AI Engineering Track**

A hands-on deep dive into NumPy — the numerical backbone of every major ML library. Covers array creation, broadcasting, vectorised operations, linear algebra, and a from-scratch implementation of the two most common feature scaling techniques used in ML preprocessing pipelines.

---

## 📋 Topics Covered

| # | Topic |
|---|-------|
| 1 | Array creation — `np.array`, `arange`, `linspace`, `zeros`, `ones`, `eye`, `random` |
| 2 | Shape inspection — `shape`, `ndim`, `dtype`, `size`, `itemsize`, `nbytes` |
| 3 | Indexing & slicing — 2D arrays, Boolean masking, fancy indexing |
| 4 | Broadcasting — row/column vector addition, outer products |
| 5 | Vectorised operations & universal functions — `sin`, `exp`, `log` |
| 6 | Aggregation — `sum`, `mean`, `std`, `min`, `max` along axes |
| 7 | Linear algebra — matrix multiply (`@`), transpose, determinant, inverse, eigenvalues |
| 8 | Feature normalisation — Min-Max Scaling & Z-Score Standardisation from scratch |

---

## 🚀 How to Run

```bash
pip install numpy
python "Week 1/Day 4/numpy_basics.py"
```

---

## 🧩 Script Structure

```
numpy_basics.py
│
├── demo_array_creation()         # 8 array types with shapes and dtypes
├── demo_shape_info()             # reshape, flatten, ndim, itemsize
├── demo_indexing()               # 2D slicing, boolean mask, fancy index
├── demo_broadcasting()           # row/col broadcast, outer product
├── demo_vectorised_ops()         # ufuncs, axis-wise aggregation
├── demo_linear_algebra()         # @, .T, det, inv, eig
└── demo_feature_normalisation()  # Min-Max & Z-Score from scratch
```

---

## 💡 Key Concepts Demonstrated

### Array Creation
```python
a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
b = np.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
c = np.linspace(0, 1, 6)          # 6 points evenly spaced
d = np.zeros((3, 4))
e = np.eye(4)                      # 4×4 identity matrix

rng = np.random.default_rng(seed=42)
f = rng.random((3, 3))             # reproducible random array
```

### Boolean Masking
```python
matrix = np.array([[10, 20, 70], [50, 60, 80]])
mask = matrix > 60
print(matrix[mask])   # [70, 80]
```

### Broadcasting
```python
matrix = np.arange(12).reshape(3, 4)  # (3, 4)
bias   = np.array([1., 2., 3., 4.])   # (4,)  — broadcast across rows
result = matrix + bias                 # no loop needed
```

### Matrix Operations
```python
A = np.array([[2, 1], [5, 3]], dtype=float)
B = np.array([[1, 2], [3, 4]], dtype=float)

C          = A @ B                  # matrix multiply
A_inv      = np.linalg.inv(A)
eigenvals  = np.linalg.eig(A)[0]
det_A      = np.linalg.det(A)
```

### Feature Normalisation (from scratch)
```python
# Min-Max Scaling → range [0, 1]
X_minmax = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Z-Score Standardisation → mean=0, std=1
X_zscore = (X - X.mean(axis=0)) / X.std(axis=0)
```
> This is exactly what `sklearn.preprocessing.MinMaxScaler` and `StandardScaler` do internally.

---

## 📊 Sample Output

```
── Feature Normalisation ──
Raw X:
         Age  Salary(k)  Experience   Score
[0]     25.0       45.0         2.0    72.0
[1]     32.0       85.0         8.0    88.0

Min-Max Scaled X (range [0,1]):
[0]   0.1304       0.0842      0.0526  0.3429
[1]   0.4348       0.5616      0.3684  0.8000

Z-Score Standardised X (mean≈0, std≈1):
[0]  -1.1310      -1.2247     -1.0911 -0.9285
[1]   0.1619       0.2449      0.1819  0.7737

── Linear Algebra ──
  A @ B:
  [[ 5.  8.]
   [14. 22.]]
  det(A) = 1.00
  Eigenvalues: [4.791  0.209]
```

---

## 🛠️ Skills Demonstrated

`NumPy` · `Vectorised Computing` · `Linear Algebra` · `Feature Engineering` · `Data Preprocessing` · `Broadcasting` · `ML Fundamentals`

---

## 👤 About Me

**Fatir Faraz** — Final-year Software Engineering student specializing in AI & Automation Engineering.
Currently interning at **Al Aziz Technologies** on the AI Engineering track, building hands-on experience across Python, ML pipelines, and AI agent development.

- 🔗 GitHub: [@Fatir002](https://github.com/Fatir002)
- 💼 LinkedIn: [linkedin.com/in/fatirfaraz](https://linkedin.com/in/fatirfaraz)
- 🌐 Fiverr: AI Automation & WhatsApp Agent Services
