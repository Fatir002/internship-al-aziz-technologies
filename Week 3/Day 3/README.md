# Day 3 – Neural Network Training

## What this covers
The complete, real-world training workflow used in almost every deep
learning project: splitting data, training in batches, choosing an
optimizer, watching for overfitting, using dropout/regularization, saving
checkpoints, and visualizing progress.

## File
- `train_neural_network.py` – Trains a small classifier on a synthetic
  "two rings" dataset (an inner ring = class 0, an outer ring = class 1).
  This shape isn't linearly separable, so it's a good test of a real neural
  network training loop.

## How to run
```bash
python train_neural_network.py
```

## What you'll see
- Training/validation loss and accuracy printed every 10 epochs.
- `best_model.pth` — the model's weights saved automatically whenever the
  validation loss improves (a model **checkpoint**).
- `training_history.png` — a plot of loss and accuracy over time for both
  the training and validation sets.

## Key concepts explained in the code
| Concept | Where it appears |
|---|---|
| Train/validation split | 80/20 split near the top of the file |
| Batch processing | `DataLoader(..., batch_size=32, ...)` |
| Optimizer (Adam) | `torch.optim.Adam(...)` — comment shows how to switch to SGD |
| Epochs | `epochs = 60` |
| Dropout / regularization | `nn.Dropout(dropout_rate)` inside `RingClassifier` |
| Validation loop | `run_one_epoch(..., optimizer=None)` |
| Overfitting detection | Compare `train_loss` vs `val_loss` in `training_history.png` |
| Model checkpoints | `if val_loss < best_val_loss: torch.save(...)` |
| Training visualization | `plot_training_history()` |

## How to spot overfitting in the chart
If training loss keeps falling while validation loss starts climbing back
up, the model is memorizing the training data instead of generalizing.
Try setting `dropout_rate=0.0` and increasing `epochs` to intentionally
cause overfitting, then compare the chart to the default run.

## Note
`best_model.pth` and `training_history.png` are generated when you run the
script — they aren't included in the repo since they're build artifacts,
not source code.
