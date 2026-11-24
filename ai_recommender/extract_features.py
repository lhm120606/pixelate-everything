"""
Feature extraction for the Pixelate parameter recommender.
=========================================================
Turns an input image into a compact, physically-meaningful feature
vector.  These features are the model INPUT.  They are deliberately kept
consistent with `dataset_classification/dataset_classifier.py` (edge
density + CIELAB colour complexity) and extended with a few texture /
resolution descriptors that also influence a good pixel-art conversion.

The exact same function is used both when generating training labels and
at inference time, so training and deployment never drift apart.
"""

import cv2
import numpy as np

# The order here defines the model's input feature layout. Keep it stable.
FEATURE_NAMES = [
    "edge_density",       # Canny edge fraction  -> detail level -> block size
    "color_complexity",   # var(a)+var(b) in LAB -> colour richness -> palette K
    "local_variance",     # mean local variance  -> texture busyness
    "mean_gradient",      # mean Sobel magnitude -> overall sharpness
    "short_side",         # min(h, w)            -> resolution scale
    "aspect_ratio",       # w / h                -> shape
]


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    img : np.ndarray
        BGR image as read by cv2.imread.

    Returns
    -------
    np.ndarray, shape (len(FEATURE_NAMES),), dtype float64
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Edge density (Canny) — same as the classifier, with median denoise
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    # 2. Colour complexity — variance of a/b channels in CIELAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)
    color_complexity = float(np.var(a) + np.var(b))

    # 3. Local variance — texture busyness (5x5 window)
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (5, 5))
    sq_mean = cv2.blur(gray_f ** 2, (5, 5))
    local_variance = float(np.mean(np.maximum(sq_mean - mean ** 2, 0.0)))

    # 4. Mean gradient magnitude (Sobel)
    gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1, ksize=3)
    mean_gradient = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    # 5-6. Geometry
    short_side = float(min(h, w))
    aspect_ratio = float(w) / float(h)

    return np.array([
        edge_density,
        color_complexity,
        local_variance,
        mean_gradient,
        short_side,
        aspect_ratio,
    ], dtype=np.float64)


def extract_features_from_path(img_path: str):
    """Convenience wrapper. Returns None on unreadable images."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return extract_features(img)
