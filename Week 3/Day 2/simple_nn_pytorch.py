"""
Day 2 - PyTorch Fundamentals (Part 2: Building a Neural Network)
--------------------------------------------------------------------
Goal: Build the SAME kind of network we hand-built in Day 1 (with raw NumPy),
but this time using PyTorch's building blocks: nn.Module, Datasets,
DataLoaders, loss functions, optimizers, and a training loop.

Concepts covered:
- Datasets & DataLoaders
- nn.Module (defining a model)
- Loss functions
- Optimizers
- Training loop
- Saving and loading models
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)



# 1. Dataset - wraps our raw data so PyTorch can loop over it in batches

class XORDataset(Dataset):
    """The XOR truth table, wrapped as a PyTorch Dataset."""

    def __init__(self):
        self.X = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        )
        self.y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 2. Model - defined as a class that inherits from nn.Module

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=8, output_size=1):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden_layer(x))
        x = self.sigmoid(self.output_layer(x))
        return x


def train_model():
    dataset = XORDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SimpleNeuralNetwork()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    epochs = 4000
    print("Training PyTorch neural network on XOR...\n")

    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()              # clear old gradients
            predictions = model(batch_X)       # forward pass
            loss = loss_function(predictions, batch_y)
            loss.backward()                    # backpropagation (automatic!)
            optimizer.step()                   # gradient descent step

        if epoch % 800 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    print(f"\nFinal loss: {loss.item():.6f}")
    return model, dataset


def evaluate_model(model, dataset):
    print("\nPredictions after training:")
    model.eval()  # switch to evaluation mode
    with torch.no_grad():  # no need to track gradients here
        for i in range(len(dataset)):
            x, actual = dataset[i]
            prediction = model(x.unsqueeze(0))  # add batch dimension
            rounded = round(prediction.item())
            print(f"Input: {x.tolist()} -> Predicted: {prediction.item():.4f} "
                  f"(rounded: {rounded}) | Actual: {int(actual.item())}")


if __name__ == "__main__":
    model, dataset = train_model()
    evaluate_model(model, dataset)

    # --- Saving and loading models ---
    save_path = "xor_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to '{save_path}'")

    loaded_model = SimpleNeuralNetwork()
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval()
    print("Model successfully loaded back from disk.")
