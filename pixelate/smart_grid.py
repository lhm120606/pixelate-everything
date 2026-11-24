"""
Unified detail-weighted grid estimation.
=======================================

All three user modes use the same estimator:

    detail map
    -> per-pixel detail weights
    -> weighted local SSIM and weighted reconstruction error
    -> largest acceptable block size

The modes differ only in how strongly high-detail pixels are weighted.
There is no separate fast heuristic and no legacy global-SSIM path.

Modes
-----
preserve
    Gives the strongest preference to high-detail regions.

balanced
    Still protects detailed regions, but gives smooth regions more influence.

simplify
    Uses the weakest detail preference of the three modes. It may choose a
    larger block, but it still uses the same weighted quality calculation and
    the same conservative candidate range as the other modes.

Public API
----------
estimate_block_size(image, mode="balanced")
compute_detail_map(image)
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np


DetailMode = Literal["preserve", "balanced", "simplify"]


# Only the spatial weighting policy changes between modes. Candidate generation,
# quality scoring, and selection tolerance are shared by all modes.
_MODE_WEIGHTS = {
    "preserve": {
        "weight_floor": 0.20,
        "weight_gain": 4.50,
        "weight_power": 1.10,
    },
    "balanced": {
        "weight_floor": 0.28,
        "weight_gain": 4.10,
        "weight_power": 1.08,
    },
    "simplify": {
        "weight_floor": 0.40,
        "weight_gain": 3.50,
        "weight_power": 1.02,
    },
}

# These values are intentionally shared. They preserve the verified conservative
# behavior of the previous preserve mode while preventing balanced/simplify from
# changing the search range or acceptable quality loss too aggressively.
_QUALITY_TOLERANCE = 0.015
_MAE_WEIGHT = 0.45
_CANDIDATE_UPPER_SCALE = 0.78
_DETAIL_UPPER_REDUCTION = 0.48


def _get_weight_profile(mode: DetailMode) -> dict[str, float]:
    """Validate and return one mode's detail-weight profile."""
    if mode not in _MODE_WEIGHTS:
        choices = ", ".join(_MODE_WEIGHTS)
        raise ValueError(f"mode must be one of: {choices}.")
    return _MODE_WEIGHTS[mode]


def _as_gray(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit grayscale image."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError("Input must be grayscale or BGR with shape (H, W, 3).")


def _robust_normalize(values: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Normalize a non-negative map using a robust upper percentile."""
    array = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    scale = float(np.percentile(array, percentile))

    if scale <= 1e-9:
        return np.zeros_like(array, dtype=np.float32)

    return np.clip(array / scale, 0.0, 1.0).astype(np.float32)


def compute_detail_map(image: np.ndarray) -> np.ndarray:
    """
    Return a normalized high-frequency detail map in [0, 1].

    Smooth gradients have a non-zero first derivative, but little high-frequency
    structure. The map therefore emphasizes high-pass residual, Laplacian
    response, local variance, and localized edge context rather than raw gradient
    magnitude alone.
    """
    gray = _as_gray(image)
    gray_f = gray.astype(np.float32)

    smooth = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=2.0, sigmaY=2.0)
    high_pass = np.abs(gray_f - smooth)

    laplacian = np.abs(cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3))

    local_mean = cv2.blur(gray_f, (7, 7))
    local_square_mean = cv2.blur(gray_f * gray_f, (7, 7))
    local_variance = np.maximum(
        local_square_mean - local_mean * local_mean,
        0.0,
    )

    blurred_u8 = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred_u8, 50, 150).astype(np.float32) / 255.0
    edge_context = cv2.GaussianBlur(edges, (0, 0), sigmaX=1.2, sigmaY=1.2)

    detail = (
        0.35 * _robust_normalize(high_pass)
        + 0.30 * _robust_normalize(laplacian)
        + 0.25 * _robust_normalize(local_variance)
        + 0.10 * _robust_normalize(edge_context)
    )

    return np.clip(detail, 0.0, 1.0).astype(np.float32)


def _detail_signal(detail_map: np.ndarray) -> float:
    """
    Summarize local detail without allowing a large smooth background to win.

    The upper-detail mean protects a small detailed subject. Coverage prevents
    a single isolated noise pixel from controlling the candidate range.
    """
    flat = np.asarray(detail_map, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0

    threshold = float(np.quantile(flat, 0.80))
    top_values = flat[flat >= threshold]
    top_mean = float(np.mean(top_values)) if top_values.size else 0.0

    coverage = float(np.mean(flat >= 0.35))
    coverage_term = min(1.0, coverage * 4.0)

    return float(np.clip(0.72 * top_mean + 0.28 * coverage_term, 0.0, 1.0))


def _detail_weights(detail_map: np.ndarray, mode: DetailMode) -> np.ndarray:
    """Build positive per-pixel weights for the requested preference mode."""
    profile = _get_weight_profile(mode)
    detail = np.clip(detail_map, 0.0, 1.0).astype(np.float64)

    weights = (
        profile["weight_floor"]
        + profile["weight_gain"]
        * np.power(detail, profile["weight_power"])
    )

    return weights.astype(np.float64)


def _pixelate(image: np.ndarray, block_size: int) -> np.ndarray:
    """Block-average pixelation resized back to source dimensions."""
    height, width = image.shape[:2]
    small_width = max(1, width // int(block_size))
    small_height = max(1, height // int(block_size))

    small = cv2.resize(
        image,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.resize(
        small,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def _ssim_map(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    """Return the per-pixel grayscale SSIM map."""
    constant_1 = (0.01 * 255.0) ** 2
    constant_2 = (0.03 * 255.0) ** 2

    image_a = gray_a.astype(np.float64)
    image_b = gray_b.astype(np.float64)

    mean_a = cv2.GaussianBlur(image_a, (11, 11), 1.5)
    mean_b = cv2.GaussianBlur(image_b, (11, 11), 1.5)

    mean_a_squared = mean_a * mean_a
    mean_b_squared = mean_b * mean_b
    mean_ab = mean_a * mean_b

    variance_a = (
        cv2.GaussianBlur(image_a * image_a, (11, 11), 1.5)
        - mean_a_squared
    )
    variance_b = (
        cv2.GaussianBlur(image_b * image_b, (11, 11), 1.5)
        - mean_b_squared
    )
    covariance = (
        cv2.GaussianBlur(image_a * image_b, (11, 11), 1.5)
        - mean_ab
    )

    numerator = (
        (2.0 * mean_ab + constant_1)
        * (2.0 * covariance + constant_2)
    )
    denominator = (
        (mean_a_squared + mean_b_squared + constant_1)
        * (variance_a + variance_b + constant_2)
    )

    return numerator / (denominator + 1e-12)


def _candidate_range(
    image: np.ndarray,
    detail_map: np.ndarray,
) -> tuple[int, int]:
    """
    Determine one conservative candidate range shared by all modes.

    This is intentionally the previously verified preserve-mode range. Mode
    selection no longer changes the search bounds.
    """
    height, width = image.shape[:2]
    short_side = min(height, width)

    minimum = max(2, short_side // 256)
    base_maximum = max(minimum + 1, short_side // 16)
    signal = _detail_signal(detail_map)

    scale = _CANDIDATE_UPPER_SCALE * (
        1.0 - _DETAIL_UPPER_REDUCTION * signal
    )

    maximum = int(round(base_maximum * scale))
    maximum = min(maximum, max(minimum + 1, short_side // 4))
    maximum = max(minimum + 1, maximum)

    return minimum, maximum


def _evaluate_block_sizes(
    image: np.ndarray,
    candidates: list[int],
    *,
    detail_map: np.ndarray,
    mode: DetailMode,
) -> list[dict[str, float | int]]:
    """Evaluate candidates with the shared weighted quality model."""
    gray = _as_gray(image)
    weights = _detail_weights(detail_map, mode)
    weight_total = float(np.sum(weights)) + 1e-12

    evaluations: list[dict[str, float | int]] = []

    for block_size in candidates:
        pixelated = _pixelate(image, block_size)
        pixelated_gray = _as_gray(pixelated)

        local_ssim = _ssim_map(gray, pixelated_gray)
        weighted_ssim = float(np.sum(weights * local_ssim) / weight_total)

        absolute_error = (
            np.abs(
                gray.astype(np.float64)
                - pixelated_gray.astype(np.float64)
            )
            / 255.0
        )
        weighted_mae = float(np.sum(weights * absolute_error) / weight_total)

        score = weighted_ssim - _MAE_WEIGHT * weighted_mae

        evaluations.append({
            "block_size": int(block_size),
            "score": float(score),
            "weighted_ssim": weighted_ssim,
            "weighted_mae": weighted_mae,
        })

    return evaluations


def _select_block_size(
    evaluations: list[dict[str, float | int]],
) -> int:
    """Select the largest block inside one shared conservative tolerance."""
    if not evaluations:
        raise ValueError("No block-size evaluations were provided.")

    best_score = max(float(item["score"]) for item in evaluations)
    minimum_score = best_score - _QUALITY_TOLERANCE

    acceptable = [
        int(item["block_size"])
        for item in evaluations
        if float(item["score"]) >= minimum_score
    ]

    return int(max(acceptable))


def estimate_block_size(
    image: np.ndarray,
    *,
    mode: DetailMode = "balanced",
    min_block: int | None = None,
    max_block: int | None = None,
    num_candidates: int = 20,
    debug: bool = False,
) -> int:
    """
    Estimate block size with the unified detail-weighted quality model.

    Parameters
    ----------
    image:
        BGR or grayscale source image.
    mode:
        Changes only the preference assigned to high-detail pixels.
    min_block, max_block:
        Optional manual search bounds.
    num_candidates:
        Number of approximately log-spaced candidate block sizes.
    debug:
        Display the detail map, score curve, and selected preview.
    """
    _get_weight_profile(mode)
    detail_map = compute_detail_map(image)

    lower, upper = _candidate_range(image, detail_map)

    if min_block is not None:
        lower = max(1, int(min_block))

    if max_block is not None:
        upper = max(lower + 1, int(max_block))

    candidates = sorted(set(
        np.geomspace(
            lower,
            upper,
            num=max(3, int(num_candidates)),
        ).astype(int).tolist()
    ))

    if len(candidates) < 3:
        candidates = list(range(lower, upper + 1))

    evaluations = _evaluate_block_sizes(
        image,
        candidates,
        detail_map=detail_map,
        mode=mode,
    )

    selected = _select_block_size(evaluations)

    if debug:
        _debug_plot(
            image,
            detail_map,
            evaluations,
            selected,
            mode,
        )

    return selected


def _debug_plot(
    image: np.ndarray,
    detail_map: np.ndarray,
    evaluations: list[dict[str, float | int]],
    selected: int,
    mode: DetailMode,
) -> None:
    """Display quality curve, mode weights, detail map, and selected preview."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[debug] matplotlib is unavailable; skipping plot.")
        return

    block_sizes = [int(item["block_size"]) for item in evaluations]
    scores = [float(item["score"]) for item in evaluations]
    selected_preview = _pixelate(image, selected)
    weights = _detail_weights(detail_map, mode)

    figure, axes = plt.subplots(1, 5, figsize=(26, 5))

    axes[0].plot(block_sizes, scores, "o-")
    axes[0].axvline(selected, linestyle="--", label=f"selected={selected}")
    axes[0].set_xlabel("Block size")
    axes[0].set_ylabel("Weighted quality score")
    axes[0].set_title(f"Unified grid evaluation: {mode}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].imshow(detail_map, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("High-frequency detail map")
    axes[1].axis("off")

    axes[2].imshow(weights, cmap="gray")
    axes[2].set_title(f"Detail weights: {mode}")
    axes[2].axis("off")

    if image.ndim == 3:
        axes[3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[4].imshow(cv2.cvtColor(selected_preview, cv2.COLOR_BGR2RGB))
    else:
        axes[3].imshow(image, cmap="gray")
        axes[4].imshow(selected_preview, cmap="gray")

    axes[3].set_title("Original")
    axes[3].axis("off")
    axes[4].set_title(f"Selected block size: {selected}")
    axes[4].axis("off")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test unified detail-weighted grid estimation."
    )
    parser.add_argument("image")
    parser.add_argument(
        "--mode",
        choices=["preserve", "balanced", "simplify"],
        default="balanced",
    )
    parser.add_argument("--min_block", type=int, default=None)
    parser.add_argument("--max_block", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()

    source = cv2.imread(arguments.image, cv2.IMREAD_COLOR)
    if source is None:
        raise SystemExit(f"Cannot read image: {arguments.image}")

    result = estimate_block_size(
        source,
        mode=arguments.mode,
        min_block=arguments.min_block,
        max_block=arguments.max_block,
        num_candidates=arguments.candidates,
        debug=arguments.debug,
    )

    print(f"Recommended block size: {result}")
