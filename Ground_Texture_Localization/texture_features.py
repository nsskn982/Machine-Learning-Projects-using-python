"""Texture feature extraction utilities.

The goal is to keep this lightweight and explainable:
- Local Binary Patterns (LBP) capture micro-texture.
- A small HSV color histogram captures coarse color differences.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_lbp(gray: np.ndarray, radius: int = 1, neighbors: int = 8) -> np.ndarray:
    """Compute a basic LBP code image.

    This implementation uses a simple (non-uniform) LBP where each neighbor
    compares against the center pixel.
    """
    # Pad the image so we can compute neighborhoods at the borders.
    padded = cv2.copyMakeBorder(gray, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    lbp = np.zeros_like(gray, dtype=np.uint8)

    # Sample neighbors on a circle.
    for n in range(neighbors):
        theta = 2.0 * np.pi * n / neighbors
        y = int(round(radius * np.sin(theta)))
        x = int(round(radius * np.cos(theta)))

        # Center pixels are compared to the neighbor pixels.
        neighbor = padded[radius + y : radius + y + gray.shape[0], radius + x : radius + x + gray.shape[1]]
        lbp |= ((neighbor >= gray) << n).astype(np.uint8)

    return lbp


def extract_features(bgr: np.ndarray) -> np.ndarray:
    """Extract a compact feature vector from a BGR image.

    Returns a 1D vector containing:
    - LBP histogram (256 bins for 8-neighbor LBP)
    - HSV color histogram (4x4x4 bins)
    """
    # Downsample for speed and to make features consistent across inputs.
    resized = cv2.resize(bgr, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    lbp = compute_lbp(gray)
    # LBP histogram encodes local texture micro-patterns.
    lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()

    # HSV histogram captures coarse color/illumination cues.
    hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    return np.concatenate([lbp_hist, hsv_hist]).astype(np.float32)
