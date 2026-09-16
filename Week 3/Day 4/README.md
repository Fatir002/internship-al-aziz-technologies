# Day 4 – Computer Vision Fundamentals

## What this covers
How images are represented as numeric data, standard preprocessing steps
(resizing, normalization, grayscale), edge detection with OpenCV, and
building your first Convolutional Neural Network (CNN) for image
classification.

## Files
- `image_processing_opencv.py` – Generates a simple image with OpenCV and
  walks through resizing, grayscale conversion, normalization, and edge
  detection (Canny).
- `image_classifier_cnn.py` – Generates a synthetic dataset of circles vs
  squares (using OpenCV drawing + random rotation augmentation), then
  trains a small CNN in PyTorch to classify them.

## Why generate images instead of downloading a dataset?
It keeps the project self-contained (no large downloads, no licensing
concerns), trains in seconds on a CPU, and makes it very easy to see
exactly what data the model is learning from.

## How to run
```bash
python image_processing_opencv.py
python image_classifier_cnn.py
```

## What you'll see
- `image_processing_opencv.py` prints image shapes/values at each
  processing step and saves the resulting images into `output_images/`.
- `image_classifier_cnn.py` prints the training loss each epoch, then
  reports test accuracy (typically 90%+) and saves the trained model to
  `shape_classifier.pth`.

## Key concepts explained in the code
| Concept | Where it appears |
|---|---|
| Images as arrays of numbers | `image.shape`, `image[0, 0]` |
| Resizing | `cv2.resize()` |
| Grayscale conversion | `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)` |
| Normalization | `image.astype(np.float32) / 255.0` |
| Edge detection | `cv2.Canny()` |
| Image augmentation | `augment()` (random rotation) |
| Convolution & filters | `nn.Conv2d` layers in `SimpleCNN` |
| Pooling | `nn.MaxPool2d` |
| Feature extraction | The stacked conv+pool layers before flattening |
| Image classification | `train_and_evaluate()` |

## How the CNN "sees" an image
`conv1` learns simple patterns (edges, curves). `conv2` combines those into
more complex shapes. `MaxPool2d` shrinks the image while keeping the
strongest signals. By the time the flattened features reach the fully
connected layers, the network has effectively learned "this pattern of
features = square" or "= circle".

## Note
`output_images/` and `shape_classifier.pth` are generated when you run the
scripts — they aren't included in the repo since they're build artifacts.
