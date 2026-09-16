"""
Day 1 - Deep Learning Fundamentals
-----------------------------------
Goal: Understand what actually happens inside a neural network by building
one from scratch using ONLY NumPy (no PyTorch/TensorFlow).

We build a tiny neural network that learns the XOR function.
XOR is a classic beginner example because it CANNOT be solved by a single
layer (linear) model - it needs at least one hidden layer, which is why it's
a good demo of why hidden layers matter.

Concepts covered:
- Artificial neurons, weights, bias
- Layers (input, hidden, output)
- Activation functions (sigmoid)
- Forward propagation
- Loss function (Mean Squared Error)
- Backpropagation (the chain rule, by hand)
- Gradient descent (manually updating weights)
- Epochs / learning rate
"""

import numpy as np

np.random.seed(42)  # reproducible results


def sigmoid(x):
    """Squashes any number into the range (0, 1)."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid_output):
    """Derivative of sigmoid, expressed in terms of its own output."""
    return sigmoid_output * (1 - sigmoid_output)


# Training data: the XOR truth table
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
y = np.array([[0], [1], [1], [0]])

# Network architecture: 2 inputs -> 4 hidden neurons -> 1 output neuron
input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.5
epochs = 10000


def forward_propagation(X):
    """Pass inputs through the network, keeping intermediate values
    for use in backpropagation."""
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def compute_loss(y_true, y_pred):
    """Mean Squared Error - how far off predictions are, on average."""
    return np.mean((y_true - y_pred) ** 2)


def backward_propagation(X, y, a1, a2):
    """Compute gradients using the chain rule and update weights
    with gradient descent."""
    global W1, b1, W2, b2
    m = X.shape[0]

    error_output = a2 - y
    d_output = error_output * sigmoid_derivative(a2)
    dW2 = a1.T @ d_output / m
    db2 = np.sum(d_output, axis=0, keepdims=True) / m

    error_hidden = d_output @ W2.T
    d_hidden = error_hidden * sigmoid_derivative(a1)
    dW1 = X.T @ d_hidden / m
    db1 = np.sum(d_hidden, axis=0, keepdims=True) / m

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


def train():
    print("Training a tiny neural network to learn XOR...\n")
    loss = None
    for epoch in range(epochs):
        _, a1, _, a2 = forward_propagation(X)
        loss = compute_loss(y, a2)
        backward_propagation(X, y, a1, a2)
        if epoch % 2000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")
    print(f"\nFinal loss: {loss:.6f}")


def evaluate():
    _, _, _, predictions = forward_propagation(X)
    print("\nFinal predictions after training:")
    for inputs, pred, actual in zip(X, predictions, y):
        rounded = round(float(pred[0]))
        print(f"Input: {inputs} -> Predicted: {pred[0]:.4f} "
              f"(rounded: {rounded}) | Actual: {actual[0]}")


if __name__ == "__main__":
    train()
    evaluate()
