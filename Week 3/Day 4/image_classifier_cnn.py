"""
Day 4 - Computer Vision Fundamentals (Part 2: Image Classification with a CNN)
------------------------------------------------------------------------------------
Goal: Build a basic Convolutional Neural Network (CNN) and train it to tell
the difference between two shapes: circles and squares.

Instead of downloading an external dataset, we GENERATE one ourselves using
OpenCV drawing functions. This keeps the example self-contained, fast to
run on a CPU, and perfect for understanding the full pipeline end to end.

Concepts covered:
- Image classification
- CNN concepts: convolution, filters, pooling, feature extraction
- Building a CNN with PyTorch
- Image augmentation (small random shifts/rotations to make the model
  more robust)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

IMG_SIZE = 32  # small images keep training fast on CPU



# 1. Generate a synthetic dataset: circles (label 0) vs squares (label 1)

def draw_circle():
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    center = (IMG_SIZE // 2 + np.random.randint(-4, 4),
              IMG_SIZE // 2 + np.random.randint(-4, 4))
    radius = np.random.randint(8, 13)
    cv2.circle(img, center, radius, 255, thickness=-1)
    return img


def draw_square():
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    size = np.random.randint(16, 24)
    x = np.random.randint(2, IMG_SIZE - size - 2)
    y = np.random.randint(2, IMG_SIZE - size - 2)
    cv2.rectangle(img, (x, y), (x + size, y + size), 255, thickness=-1)
    return img


def augment(img):
    """Simple image augmentation: random small rotation, so the model
    doesn't just memorize exact pixel positions."""
    angle = np.random.uniform(-20, 20)
    matrix = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (IMG_SIZE, IMG_SIZE))


def build_dataset(n_per_class=300):
    images, labels = [], []
    for _ in range(n_per_class):
        images.append(augment(draw_circle()))
        labels.append(0)
        images.append(augment(draw_square()))
        labels.append(1)

    images = np.array(images, dtype=np.float32) / 255.0   # normalize to [0, 1]
    images = images[:, np.newaxis, :, :]                   # add channel dim: (N, 1, H, W)
    labels = np.array(labels, dtype=np.int64)

    X = torch.from_numpy(images)
    y = torch.from_numpy(labels)
    return X, y



# 2. A basic CNN

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution layer 1: learns simple features (edges, curves)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        # Convolution layer 2: learns combinations of those simple features
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # shrinks the image, keeps key features
        self.relu = nn.ReLU()

        # After 2 pooling layers, a 32x32 image becomes 8x8
        self.fc1 = nn.Linear(16 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)  # 2 output classes: circle, square

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # -> (N, 8, 16, 16)
        x = self.pool(self.relu(self.conv2(x)))  # -> (N, 16, 8, 8)
        x = x.view(x.size(0), -1)                # flatten for the fully-connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)                          # raw scores (logits) - no softmax needed here
        return x


def train_and_evaluate():
    X, y = build_dataset(n_per_class=300)

    # 80/20 train/test split
    n_total = len(X)
    n_train = int(0.8 * n_total)
    perm = torch.randperm(n_total)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X[test_idx], y[test_idx]), batch_size=32)

    model = SimpleCNN()
    loss_function = nn.CrossEntropyLoss()  # standard loss for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print("Training a CNN to classify circles vs squares...\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:2d}/{epochs} | Avg training loss: {total_loss/len(train_loader):.4f}")

    # --- Evaluate on the held-out test set ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    print(f"\nTest accuracy: {correct/total:.2%}  ({correct}/{total} correct)")
    torch.save(model.state_dict(), "shape_classifier.pth")
    print("Model saved to 'shape_classifier.pth'")


if __name__ == "__main__":
    train_and_evaluate()
