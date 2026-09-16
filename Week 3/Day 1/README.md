# Day 1 – Deep Learning Fundamentals

## What this covers
An introduction to how neural networks actually work under the hood:
artificial neurons, layers, weights, bias, activation functions, forward
propagation, loss functions, backpropagation, and gradient descent.

## File
- `neural_network_from_scratch.py` – A neural network built **from scratch
  using only NumPy** (no PyTorch/TensorFlow) that learns the XOR logic gate.

## Why XOR?
XOR is the classic "hello world" of deep learning because it **cannot** be
solved by a simple linear model — it needs at least one hidden layer. This
makes it a perfect small example to show why hidden layers exist and how
backpropagation adjusts weights to solve a non-linear problem.

## How to run
```bash
python neural_network_from_scratch.py
```

## What you'll see
The script prints the training loss every 2000 epochs (it should keep
decreasing), and at the end shows the network's predictions for all 4 XOR
input combinations compared against the correct answers.

## Key concepts explained in the code
| Concept | Where it appears |
|---|---|
| Weights & bias | `W1, b1, W2, b2` |
| Forward propagation | `forward_propagation()` |
| Activation function (sigmoid) | `sigmoid()` |
| Loss function (MSE) | `compute_loss()` |
| Backpropagation (chain rule) | `backward_propagation()` |
| Gradient descent | weight update lines (`W1 -= learning_rate * dW1`, etc.) |
| Epochs / learning rate | `epochs`, `learning_rate` variables |

## Takeaway
Frameworks like PyTorch automate all of this math for you, but understanding
it manually first makes everything in the following days (PyTorch tensors,
`nn.Module`, `loss.backward()`, optimizers) make a lot more sense.
