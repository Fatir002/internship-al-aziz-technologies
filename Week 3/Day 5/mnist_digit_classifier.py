"""
Day 5 - Weekly Project: Handwritten Digit Classifier (MNIST)
------------------------------------------------------------------
This project combines everything learned in Week 3:
  - Deep learning fundamentals (Day 1)
  - PyTorch basics: tensors, nn.Module, training loop (Day 2)
  - A full training workflow: train/val split, optimizer, checkpoints,
    training visualization (Day 3)
  - Computer vision: image preprocessing & CNNs (Day 4)

We train a Convolutional Neural Network on the MNIST dataset (70,000
images of handwritten digits, 0-9) - one of the most famous beginner
datasets in computer vision.

NOTE: The first time you run this, PyTorch will automatically DOWNLOAD the
MNIST dataset (~10MB) into a local "./data" folder, so you'll need an
active internet connection for that first run only.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# 1. Load and preprocess the MNIST dataset

# transforms.ToTensor() converts images (0-255) to tensors scaled to [0, 1]
# transforms.Normalize(...) further standardizes pixel values using MNIST's
# known mean/std - a common preprocessing step for image models.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Split the training set into train/validation (90/10)
n_val = int(0.1 * len(full_train_dataset))
n_train = len(full_train_dataset) - n_val
train_dataset, val_dataset = random_split(full_train_dataset, [n_train, n_val])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

print(f"Training samples: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")


# 2. Define the CNN

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2, 2)                            # halves image size
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)                           # reduces overfitting
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)  # 10 output classes: digits 0-9

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # -> (N, 16, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # -> (N, 32, 7, 7)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def run_epoch(model, loader, loss_function, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = loss_function(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train():
    model = DigitCNN().to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5  # MNIST is easy - a CNN reaches ~99% accuracy in just a few epochs
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print("\nTraining CNN on MNIST...\n")
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, loss_function, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_function, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "mnist_cnn_best.pth")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

    print(f"\nBest validation accuracy: {best_val_acc:.2%} (model saved to 'mnist_cnn_best.pth')")
    return model, history


def evaluate_on_test_set(model):
    loss_function = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(model, test_loader, loss_function, optimizer=None)
    print(f"\nFinal test set performance -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")
    return test_acc


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Accuracy")
    axes[1].plot(history["val_acc"], label="Validation Accuracy")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("mnist_training_history.png")
    print("Training curves saved to 'mnist_training_history.png'")


def show_sample_predictions(model, n=8):
    """Saves a grid of test images with the model's predicted label vs
    the true label - a nice visual sanity check."""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:n].to(DEVICE), labels[:n]

    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu()

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    for i in range(n):
        img = images[i].cpu().squeeze() * 0.3081 + 0.1307  # undo normalization
        axes[i].imshow(img, cmap="gray")
        color = "green" if predictions[i] == labels[i] else "red"
        axes[i].set_title(f"Pred: {predictions[i].item()}", color=color)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("mnist_sample_predictions.png")
    print("Sample predictions saved to 'mnist_sample_predictions.png'")


if __name__ == "__main__":
    trained_model, history = train()
    evaluate_on_test_set(trained_model)
    plot_history(history)
    show_sample_predictions(trained_model)
