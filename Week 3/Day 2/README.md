# Day 2 – PyTorch Fundamentals

## What this covers
Moving from raw NumPy (Day 1) to PyTorch — the framework most modern AI
engineers use to build and train neural networks. PyTorch handles tensor
math, GPU acceleration, and gradient computation for you.

## Files
- `pytorch_tensor_basics.py` – Introduction to PyTorch tensors: creating
  them, basic math operations, reshaping, indexing, matrix multiplication,
  GPU vs CPU, and a first look at **autograd** (automatic differentiation).
- `simple_nn_pytorch.py` – Rebuilds the same XOR problem from Day 1, but
  using PyTorch's `Dataset`, `DataLoader`, `nn.Module`, a real training
  loop, an optimizer, and model saving/loading.

## How to run
```bash
python pytorch_tensor_basics.py
python simple_nn_pytorch.py
```

## What you'll see
- `pytorch_tensor_basics.py` prints tensor operations and confirms autograd
  correctly computes a derivative by hand.
- `simple_nn_pytorch.py` trains a small network until the loss is close to
  zero, prints its predictions on the XOR truth table, then saves the
  trained weights to `xor_model.pth` and reloads them to prove it works.

## Key concepts explained in the code
| Concept | Where it appears |
|---|---|
| Tensors | `pytorch_tensor_basics.py` |
| Autograd | `y.backward()` in `pytorch_tensor_basics.py` |
| GPU vs CPU | `torch.cuda.is_available()` |
| Dataset / DataLoader | `XORDataset`, `DataLoader` in `simple_nn_pytorch.py` |
| nn.Module | `SimpleNeuralNetwork` class |
| Loss function | `nn.MSELoss()` |
| Optimizer | `torch.optim.Adam` |
| Training loop | `train_model()` |
| Saving/loading models | `torch.save()` / `torch.load()` |

## Comparing Day 1 vs Day 2
Day 1 required manually deriving and coding every gradient by hand. Day 2
does the exact same job in a fraction of the code because `loss.backward()`
computes all gradients automatically, and `optimizer.step()` applies them.
This is why PyTorch (and frameworks like it) are the industry standard.

## Note
`xor_model.pth` is generated when you run the script — it's not included in
the repo since it's a build artifact, not source code.
