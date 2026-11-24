"""
Region segmentation for MVP ver.4.
====================================

This module segments a BGR image into visually meaningful regions using
Canny edge detection + watershed. It is intended to support region-balanced
palette estimation, so palette learning is not dominated only by large-area
pixel frequency.

Public API
----------
segment_image(image, edge_threshold)
    Returns a 2-D int32 label map. Label 0 means watershed boundary;
    labels 1..N are image regions.

visualize_segments(image, labels)
    Convenience debug visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Union

import cv2
import numpy as np


Threshold = Union[int, float, Tuple[Union[int, float], Union[int, float]]]


@dataclass(frozen=True)
class RegionStats:
    """Compact per-region statistics for downstream palette logic."""

    label: int
    area: int
    mean_bgr: np.ndarray
    mean_lab: np.ndarray


def _parse_edge_threshold(edge_threshold: Threshold) -> Tuple[int, int]:
    """
    Convert edge_threshold into Canny's (low, high) thresholds.

    Usage options:
      - edge_threshold = 80      -> low=80, high=160
      - edge_threshold = (50,150)-> low=50, high=150
    """
    if isinstance(edge_threshold, Iterable) and not isinstance(edge_threshold, (str, bytes)):
        vals = list(edge_threshold)
        if len(vals) != 2:
            raise ValueError("edge_threshold tuple/list must contain exactly two values: (low, high).")
        low, high = int(vals[0]), int(vals[1])
    else:
        low = int(edge_threshold)
        high = int(round(low * 2.0))

    low = max(0, min(low, 255))
    high = max(low + 1, min(high, 255))
    return low, high



def compute_edge_mask(
    image: np.ndarray,
    edge_threshold: Threshold = 80,
    *,
    blur_ksize: int = 5,
) -> np.ndarray:
    """Return the raw uint8 Canny mask used by segmentation."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "compute_edge_mask expects BGR image with shape (H, W, 3)."
        )

    blur_ksize = max(1, int(blur_ksize))
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    low, high = _parse_edge_threshold(edge_threshold)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 1:
        gray = cv2.medianBlur(gray, blur_ksize)

    return cv2.Canny(gray, low, high)



def _remove_small_regions(labels: np.ndarray, min_region_size: int) -> np.ndarray:
    """
    Remove tiny watershed regions by marking them as boundary label 0.

    This keeps the module conservative: tiny fragments are not allowed to
    dominate region-count-based palette estimation.
    """
    if min_region_size <= 1:
        return labels

    cleaned = labels.copy()
    region_ids, counts = np.unique(cleaned[cleaned > 0], return_counts=True)
    small_ids = region_ids[counts < min_region_size]
    if small_ids.size > 0:
        cleaned[np.isin(cleaned, small_ids)] = 0
    return cleaned


def segment_image(
    image: np.ndarray,
    edge_threshold: Threshold = 80,
    *,
    blur_ksize: int = 5,
    edge_dilate_iter: int = 1,
    min_region_size: int = 24,
) -> np.ndarray:
    """
    Segment a BGR image using edge detection and watershed.

    Parameters
    ----------
    image:
        Input image in BGR format, as returned by cv2.imread.
    edge_threshold:
        Canny threshold. If a scalar is given, it is interpreted as the lower
        threshold and the upper threshold is set to 2x. If a pair is given,
        it is interpreted as (low, high).
    blur_ksize:
        Median blur kernel size before edge detection. Must be odd. Larger
        values suppress noise but may merge fine regions.
    edge_dilate_iter:
        Number of dilation iterations applied to the edge map before marker
        generation. Larger values create stronger watershed boundaries.
    min_region_size:
        Regions smaller than this many pixels are discarded into boundary
        label 0.

    Returns
    -------
    np.ndarray, shape (H, W), dtype int32
        Label map. 0 means watershed boundary / discarded tiny fragment;
        positive integers are region IDs.

    Notes
    -----
    This is deliberately not a semantic segmenter. It is a lightweight,
    transparent segmentation module for color-region balancing in pixel-art
    preprocessing.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("segment_image expects a BGR image with shape (H, W, 3).")

    blur_ksize = int(blur_ksize)
    if blur_ksize < 1:
        blur_ksize = 1
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    # 1) Edge detection: likely visual region boundaries.
    edges = compute_edge_mask(
        image,
        edge_threshold,
        blur_ksize=blur_ksize,
    )

    # 2) Strengthen edge boundaries so connected components become stable
    #    watershed markers rather than leaking across weak edges.
    kernel = np.ones((3, 3), dtype=np.uint8)
    if edge_dilate_iter > 0:
        edges = cv2.dilate(edges, kernel, iterations=int(edge_dilate_iter))

    # 3) Marker construction. Interiors are non-edge areas; boundaries are 0.
    interiors = cv2.bitwise_not(edges)
    interiors = cv2.morphologyEx(interiors, cv2.MORPH_OPEN, kernel, iterations=1)

    # connectedComponents returns labels: 0 is background, 1..N are interiors.
    n_labels, markers = cv2.connectedComponents(interiors)

    # Shift labels upward so watershed can use 0 as unknown/boundary.
    markers = markers.astype(np.int32) + 1
    markers[edges > 0] = 0

    # 4) Watershed. cv2.watershed mutates markers in-place; boundary pixels -> -1.
    ws = markers.copy()
    cv2.watershed(image, ws)

    # 5) Normalize output convention: boundaries/tiny fragments are 0, regions are 1..N.
    labels = ws.copy()
    labels[labels <= 1] = 0  # watershed boundary -1 and artificial background 1
    labels[labels > 1] -= 1
    labels = _remove_small_regions(labels.astype(np.int32), min_region_size)

    # Re-index positive labels to be contiguous: 1..N.
    positive = np.unique(labels[labels > 0])
    remapped = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(positive, start=1):
        remapped[labels == old_id] = new_id

    return remapped


def compute_region_stats(image: np.ndarray, labels: np.ndarray) -> list[RegionStats]:
    """
    Compute area and mean BGR/LAB color for every positive region label.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")
    if labels.shape[:2] != image.shape[:2]:
        raise ValueError("labels must have the same height and width as image.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    stats: list[RegionStats] = []
    for label in np.unique(labels):
        if label <= 0:
            continue
        mask = labels == label
        area = int(np.count_nonzero(mask))
        if area == 0:
            continue
        mean_bgr = image[mask].mean(axis=0).astype(np.float32)
        mean_lab = lab[mask].mean(axis=0).astype(np.float32)
        stats.append(RegionStats(int(label), area, mean_bgr, mean_lab))
    return stats


def visualize_segments(image: np.ndarray, labels: np.ndarray, *, boundary_color=(0, 0, 255)) -> np.ndarray:
    """
    Build a BGR debug image where each segment is filled by its mean color and
    watershed boundaries are highlighted.
    """
    if labels.shape[:2] != image.shape[:2]:
        raise ValueError("labels must have the same height and width as image.")

    out = np.zeros_like(image)
    for stat in compute_region_stats(image, labels):
        out[labels == stat.label] = np.clip(stat.mean_bgr, 0, 255).astype(np.uint8)
    out[labels == 0] = np.array(boundary_color, dtype=np.uint8)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--edge_threshold", type=int, default=80)
    parser.add_argument("--out", default="segments_debug.png")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    label_map = segment_image(img, args.edge_threshold)
    debug = visualize_segments(img, label_map)
    cv2.imwrite(args.out, debug)
    print(f"segments={len(np.unique(label_map[label_map > 0]))}; saved -> {args.out}")
