"""
Day 2 - PyTorch Fundamentals (Part 1: Tensors)
------------------------------------------------
Goal: Get comfortable with PyTorch's core data structure - the Tensor.
A tensor is basically a NumPy array that can also run on a GPU and
automatically track gradients for backpropagation.

Concepts covered:
- What is PyTorch?
- Creating tensors
- Tensor operations (math, reshaping, indexing)
- GPU vs CPU
"""

import torch

print("PyTorch version:", torch.__version__)


# 1. Creating tensors

print("\n--- Creating tensors ---")

t_from_list = torch.tensor([1, 2, 3, 4])
print("From a list:", t_from_list)

t_zeros = torch.zeros(2, 3)          # 2x3 tensor of zeros
t_ones = torch.ones(2, 3)            # 2x3 tensor of ones
t_random = torch.rand(2, 3)          # 2x3 tensor of random values [0, 1)

print("Zeros:\n", t_zeros)
print("Ones:\n", t_ones)
print("Random:\n", t_random)


# 2. Tensor properties

print("\n--- Tensor properties ---")
print("Shape:", t_random.shape)
print("Data type:", t_random.dtype)


# 3. Basic tensor operations (just like NumPy)

print("\n--- Basic operations ---")
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("a + b =", a + b)
print("a * b =", a * b)          # element-wise multiplication
print("a . b =", torch.dot(a, b))  # dot product
print("mean(a) =", a.mean())
print("sum(a) =", a.sum())


# 4. Reshaping and indexing

print("\n--- Reshaping & indexing ---")
matrix = torch.arange(12)          # tensor([0, 1, 2, ..., 11])
print("Original:", matrix)

reshaped = matrix.reshape(3, 4)    # turn into a 3x4 matrix
print("Reshaped to 3x4:\n", reshaped)
print("First row:", reshaped[0])
print("Element at row 1, col 2:", reshaped[1, 2].item())


# 5. Matrix multiplication (the backbone of neural networks)

print("\n--- Matrix multiplication ---")
m1 = torch.rand(2, 3)
m2 = torch.rand(3, 2)
result = m1 @ m2   # same as torch.matmul(m1, m2)
print("m1 shape:", m1.shape, "| m2 shape:", m2.shape, "| result shape:", result.shape)


# 6. GPU vs CPU

print("\n--- GPU vs CPU ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Moving a tensor to whatever device is available
t_on_device = t_random.to(device)
print("Tensor is now on:", t_on_device.device)


# 7. Autograd - PyTorch tracking gradients automatically

print("\n--- Autograd preview (used heavily in Day 3) ---")
x = torch.tensor(2.0, requires_grad=True)  # tell PyTorch to track this
y = x ** 2 + 3 * x + 1                     # y = x^2 + 3x + 1

y.backward()  # compute dy/dx automatically
print(f"For y = x^2 + 3x + 1 at x=2, dy/dx = {x.grad.item()} "
      f"(expected: 2x + 3 = {2*2 + 3})")
