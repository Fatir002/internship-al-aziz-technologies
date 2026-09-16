"""
Day 4 - Computer Vision Fundamentals (Part 1: Image Processing with OpenCV)
-------------------------------------------------------------------------------
Goal: Learn how images are represented as data (arrays of pixel values) and
practice the standard preprocessing steps every computer vision pipeline uses.

We don't need an external image file - we generate our own simple image
using OpenCV's drawing functions, which keeps this beginner-friendly and
avoids any copyright/licensing issues with downloaded images.

Concepts covered:
- Images as data (arrays of numbers)
- Image preprocessing: resizing, normalization, grayscale conversion
- Basic OpenCV usage
- Edge detection (a classic feature-extraction technique)
"""

import cv2
import numpy as np
import os

OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_sample_image():
    """Creates a simple 300x300 color image with a few shapes on it,
    so we have something to process without needing an external file."""
    image = np.full((300, 300, 3), 255, dtype=np.uint8)  # white background

    cv2.rectangle(image, (30, 30), (130, 130), (255, 0, 0), thickness=-1)   # blue square
    cv2.circle(image, (220, 80), 50, (0, 0, 255), thickness=-1)            # red circle
    cv2.line(image, (30, 220), (270, 220), (0, 255, 0), thickness=5)       # green line

    return image


def main():
    print("Images are just arrays of numbers - let's prove it.\n")

    # --- 1. Images as data ---
    image = create_sample_image()
    print(f"Image shape: {image.shape}  (height, width, color channels)")
    print(f"Image data type: {image.dtype}")
    print(f"Pixel value at (0, 0) [top-left corner]: {image[0, 0]}  (Blue, Green, Red)")
    cv2.imwrite(f"{OUTPUT_DIR}/01_original.png", image)

    # --- 2. Resizing ---
    resized = cv2.resize(image, (150, 150))  # shrink to half size
    print(f"\nResized shape: {resized.shape}")
    cv2.imwrite(f"{OUTPUT_DIR}/02_resized.png", resized)

    # --- 3. Grayscale conversion ---
    # Neural networks often work with grayscale images to reduce input size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale shape: {gray.shape}  (no color channel dimension)")
    cv2.imwrite(f"{OUTPUT_DIR}/03_grayscale.png", gray)

    # --- 4. Normalization ---
    # Pixel values normally range 0-255. Neural networks train better when
    # inputs are scaled to a small range like 0-1.
    normalized = image.astype(np.float32) / 255.0
    print(f"\nBefore normalization -> min: {image.min()}, max: {image.max()}")
    print(f"After normalization  -> min: {normalized.min():.2f}, max: {normalized.max():.2f}")

    # --- 5. Edge detection (Canny) ---
    # A classic way to extract "features" (shape outlines) from an image
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    cv2.imwrite(f"{OUTPUT_DIR}/04_edges.png", edges)
    print(f"\nEdge-detected image saved. Non-zero (edge) pixels: {np.count_nonzero(edges)}")

    print(f"\nAll processed images saved inside '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
