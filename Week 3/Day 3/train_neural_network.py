"""
Day 3 - Neural Network Training
----------------------------------
Goal: Learn the FULL training workflow used for almost every deep learning
project: train/validation split, batching, optimizers, tracking loss,
detecting overfitting, using dropout/regularization, saving checkpoints,
and visualizing training progress.

We train a small classifier on a synthetic 2D "two rings" dataset (one
class forms an inner ring, the other an outer ring) - a classic non-linear
classification problem that a simple neural network can solve.

Concepts covered:
- Training / validation split
- Optimizers (SGD and Adam)
- Epochs, batch processing
- Overfitting vs underfitting
- Dropout & regularization
- Model checkpoints (saving the BEST model, not just the last one)
- Training visualization (loss curve saved as a PNG)
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)


# 1. Create a synthetic "two rings" dataset

def make_two_rings(n_samples=1000, noise=0.1):
    """Generates points for two concentric rings - class 0 (inner ring)
    and class 1 (outer ring). This is NOT linearly separable, so it's a
    good test for a neural network with a hidden layer."""
    n_per_class = n_samples // 2

    angles_inner = torch.rand(n_per_class) * 2 * 3.14159
    radius_inner = 1.0 + torch.randn(n_per_class) * noise
    x_inner = torch.stack(
        [
            radius_inner * torch.cos(angles_inner),
            radius_inner * torch.sin(angles_inner),
        ],
        dim=1,
    )

    angles_outer = torch.rand(n_per_class) * 2 * 3.14159
    radius_outer = 2.5 + torch.randn(n_per_class) * noise
    x_outer = torch.stack(
        [
            radius_outer * torch.cos(angles_outer),
            radius_outer * torch.sin(angles_outer),
        ],
        dim=1,
    )

    X = torch.cat([x_inner, x_outer], dim=0)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])
    return X, y.unsqueeze(1)


X, y = make_two_rings(n_samples=1000)

# --- Train / validation split (80% train, 20% validation) ---
n_total = len(X)
n_train = int(0.8 * n_total)
perm = torch.randperm(n_total)
train_idx, val_idx = perm[:n_train], perm[n_train:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)


# 2. Model with dropout (to fight overfitting)

class RingClassifier(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # randomly "turns off" neurons during training
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def run_one_epoch(model, loader, loss_function, optimizer=None):
    """Runs one pass over the data. If an optimizer is given, it trains;
    if not, it just evaluates (used for validation)."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch_X, batch_y in loader:
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(batch_X)
            correct += ((predictions > 0.5).float() == batch_y).sum().item()
            total += len(batch_X)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train():
    model = RingClassifier()
    loss_function = nn.BCELoss()  # Binary Cross-Entropy for binary classification

    # Adam is used here - generally converges faster than plain SGD.
    # (Try switching to torch.optim.SGD(model.parameters(), lr=0.1) to compare!)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 60
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    checkpoint_path = "best_model.pth"

    print("Training neural network on the two-rings dataset...\n")
    for epoch in range(epochs):
        train_loss, train_acc = run_one_epoch(
            model, train_loader, loss_function, optimizer
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, loss_function, optimizer=None
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # --- Model checkpointing: only save when validation improves ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} "
                f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}"
            )

    print(
        f"\nBest model (lowest val loss = {best_val_loss:.4f}) saved to '{checkpoint_path}'"
    )
    return model, history


def plot_training_history(history):
    """Saves a PNG of the training/validation loss and accuracy curves.
    This is essential for visually spotting overfitting: if train loss keeps
    dropping while val loss starts rising, the model is overfitting."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Validation Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Training curves saved to 'training_history.png'")


if __name__ == "__main__":
    trained_model, history = train()
    plot_training_history(history)
