# Day 5 – Weekly Project: MNIST Handwritten Digit Classifier

## What this covers
The Week 3 capstone project. It combines everything from the week into one
complete, real pipeline:
- Deep learning fundamentals (Day 1)
- PyTorch tensors, `nn.Module`, and a training loop (Day 2)
- A full training workflow — train/val split, optimizer, checkpoints,
  training visualization (Day 3)
- Image preprocessing and CNNs (Day 4)

## File
- `mnist_digit_classifier.py` – Trains a Convolutional Neural Network to
  recognize handwritten digits (0–9) using the classic **MNIST** dataset
  (70,000 images), one of the most well-known benchmark datasets in
  computer vision.

## How to run
```bash
python mnist_digit_classifier.py
```

> **Note:** The first run automatically downloads the MNIST dataset
> (~10 MB) into a local `./data` folder, so you'll need an internet
> connection the first time. After that, it's cached locally.

## What you'll see
- Training/validation loss and accuracy for each of 5 epochs (should reach
  ~98–99% validation accuracy).
- Final accuracy on the held-out test set.
- `mnist_cnn_best.pth` — the best-performing model checkpoint.
- `mnist_training_history.png` — loss/accuracy curves.
- `mnist_sample_predictions.png` — a small grid of test digits with the
  model's predictions (green = correct, red = incorrect).

## Architecture summary
```
Input (1x28x28 grayscale digit)
  -> Conv2D(16 filters) -> ReLU -> MaxPool  -> 16x14x14
  -> Conv2D(32 filters) -> ReLU -> MaxPool  -> 32x7x7
  -> Flatten -> Fully Connected(64) -> Dropout -> ReLU
  -> Fully Connected(10)  [one score per digit 0-9]
```

## Why this is a good portfolio project
MNIST is a widely recognized benchmark, so a recruiter can immediately
understand what the project does and judge the reported accuracy against a
known baseline. The code also demonstrates the complete ML engineering
workflow (data loading, preprocessing, train/val/test split, training,
checkpointing, evaluation, visualization) rather than just a toy script.

## Ideas for extending this project
- Try adding more convolutional layers or Batch Normalization.
- Test the trained model on your own handwritten digit (scan/photo + resize
  to 28x28 grayscale).
- Wrap the trained model in a small Flask/FastAPI endpoint that accepts an
  image and returns the predicted digit (great segue into Week 5's API work).

## Note
`data/`, `mnist_cnn_best.pth`, and the generated `.png` files are build
artifacts created when you run the script — they aren't included in the
repo.
