"""
Region-balanced palette extraction with exact-budget compression.
================================================================

Active compressed path:

    image
    -> Canny + watershed segmentation
    -> one LAB representative per region
    -> over-clustered weighted K-means
    -> exact-budget palette compression

Segmentation is used only to generate visually balanced color samples. No
region is classified as a gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import KMeans

from compress_representatives import CompressionResult
from compress_representatives import compress_representatives
from exclude_outline import OutlineRecord
from exclude_outline import detect_outline_record
from segment_image import Threshold
from segment_image import compute_region_stats
from segment_image import segment_image


@dataclass(frozen=True)
class CompressedPaletteResult:
    """Metadata required by the exact-budget compressed renderer."""

    cluster_map: np.ndarray
    cluster_colors: np.ndarray
    cluster_weights: np.ndarray
    compression: CompressionResult
    region_labels: np.ndarray
    outline_record: OutlineRecord | None

    requested_k: int
    initial_k: int
    compressed_k: int

    @property
    def solid_group_count(self) -> int:
        return sum(group.mode == "solid" for group in self.compression.groups)

    @property
    def dither_group_count(self) -> int:
        return sum(group.mode == "dither" for group in self.compression.groups)

    @property
    def output_colors(self) -> np.ndarray:
        return self.compression.output_palette().astype(np.uint8)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def _fit_kmeans(
    samples: np.ndarray,
    k: int,
    *,
    sample_weight: np.ndarray | None = None,
    random_state: int = 42,
) -> KMeans:
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > len(samples):
        raise ValueError("k cannot exceed the number of samples.")

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=random_state,
    )
    model.fit(
        samples.astype(np.float32),
        sample_weight=sample_weight,
    )
    return model


def _assign_pixels_to_centroids_lab(
    lab_pixels: np.ndarray,
    centroids_lab: np.ndarray,
    *,
    chunk_size: int = 50000,
) -> np.ndarray:
    pixels = lab_pixels.astype(np.float32).reshape(-1, 3)
    centroids = centroids_lab.astype(np.float32).reshape(-1, 3)
    assigned = np.empty(len(pixels), dtype=np.int32)

    for start in range(0, len(pixels), chunk_size):
        stop = min(start + chunk_size, len(pixels))
        difference = pixels[start:stop, None, :] - centroids[None, :, :]
        assigned[start:stop] = np.argmin(
            np.sum(difference * difference, axis=2),
            axis=1,
        )

    return assigned


def _lab_to_bgr_colors(colors_lab: np.ndarray) -> np.ndarray:
    if len(colors_lab) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    lab = np.clip(
        np.rint(colors_lab),
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        lab.reshape(1, -1, 3),
        cv2.COLOR_LAB2BGR,
    ).reshape(-1, 3)


def _region_weights(
    areas: np.ndarray,
    mode: str,
    area_cap: int,
) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(areas), dtype=np.float64)
    if mode == "area":
        return areas.astype(np.float64)
    if mode == "capped_area":
        return np.minimum(areas.astype(np.float64), float(area_cap))

    raise ValueError(
        'weight_mode must be "uniform", "area", or "capped_area".'
    )



def _resolve_gradient_tolerance(
    gradient_tolerance: float,
    *,
    minimum_dither_span: float,
    minimum_linearity: float,
    maximum_reconstruction_rmse: float,
    maximum_point_error: float,
    minimum_endpoint_support_fraction: float,
) -> dict[str, float]:
    """
    Convert one user-facing tolerance into effective dither-family thresholds.

    gradient_tolerance = 1.0 preserves the explicit threshold values.
    Values above 1.0 make color ramps easier to accept; values below 1.0 make
    them harder to accept. Square-root scaling keeps the control gradual.
    """
    tolerance = float(gradient_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "gradient_tolerance must be a finite value greater than 0."
        )

    scale = float(np.sqrt(tolerance))

    # Scale the complement so 1.0 remains the strict upper bound.
    effective_linearity = 1.0 - (
        (1.0 - float(minimum_linearity)) * scale
    )

    return {
        "minimum_dither_span": max(
            0.0,
            float(minimum_dither_span) / scale,
        ),
        "minimum_linearity": float(np.clip(
            effective_linearity,
            0.0,
            1.0,
        )),
        "maximum_reconstruction_rmse": max(
            0.0,
            float(maximum_reconstruction_rmse) * scale,
        ),
        "maximum_point_error": max(
            0.0,
            float(maximum_point_error) * scale,
        ),
        "minimum_endpoint_support_fraction": float(np.clip(
            float(minimum_endpoint_support_fraction) / scale,
            0.0,
            0.5,
        )),
    }


def _fit_region_palette(
    img: np.ndarray,
    region_labels: np.ndarray,
    stats: list,
    *,
    initial_k: int,
    sample_weights: np.ndarray,
    random_state: int,
) -> tuple[KMeans, np.ndarray, np.ndarray, np.ndarray]:
    """Fit K-means and create a full-image cluster map."""
    region_lab = np.vstack([
        stat.mean_lab
        for stat in stats
    ]).astype(np.float32)

    model = _fit_kmeans(
        region_lab,
        initial_k,
        sample_weight=sample_weights,
        random_state=random_state,
    )

    region_clusters = model.predict(region_lab)
    cluster_map = np.zeros(region_labels.shape, dtype=np.int32)

    for stat, cluster_id in zip(stats, region_clusters):
        cluster_map[region_labels == stat.label] = int(cluster_id)

    boundary = region_labels == 0
    if np.any(boundary):
        image_lab = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)
        cluster_map[boundary] = _assign_pixels_to_centroids_lab(
            image_lab[boundary],
            model.cluster_centers_,
        )

    cluster_colors = _lab_to_bgr_colors(model.cluster_centers_)
    cluster_weights = np.bincount(
        region_clusters,
        weights=sample_weights,
        minlength=initial_k,
    ).astype(np.float64)
    cluster_weights = np.maximum(cluster_weights, 1e-12)

    return model, cluster_map, cluster_colors, cluster_weights




def _hybrid_palette_samples(
    img: np.ndarray,
    stats: list,
    region_weights: np.ndarray,
    *,
    minimum_sample_count: int,
    max_pixel_samples: int,
    pixel_weight_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build region-first palette samples with a raw-pixel fallback.

    Region means remain the primary weighted samples. Raw image colors are
    added only when segmentation provides too few representatives for the
    requested exact palette budget or over-clustered initial palette.

    Returns:
        combined LAB samples
        combined sample weights
        region LAB colors
        image LAB pixels
    """
    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat_pixels = image_lab.reshape(-1, 3)

    region_lab = np.vstack([
        stat.mean_lab
        for stat in stats
    ]).astype(np.float32)

    if len(region_lab) >= minimum_sample_count:
        return region_lab, region_weights, region_lab, flat_pixels

    unique_pixels = np.unique(flat_pixels.astype(np.uint8), axis=0).astype(np.float32)
    if len(unique_pixels) < minimum_sample_count:
        raise ValueError(
            "The image does not contain enough distinct 8-bit colors for the "
            f"requested palette construction: distinct_pixels={len(unique_pixels)}, "
            f"required={minimum_sample_count}."
        )

    rng = np.random.default_rng(random_state)
    if len(unique_pixels) > max_pixel_samples:
        chosen = rng.choice(
            len(unique_pixels),
            size=max_pixel_samples,
            replace=False,
        )
        pixel_samples = unique_pixels[chosen]
    else:
        pixel_samples = unique_pixels

    # Region samples remain dominant. Supplemental pixels collectively receive
    # only a controlled fraction of the region-sample total weight.
    region_total = float(np.sum(region_weights))
    supplemental_total = max(
        1e-6,
        region_total * float(pixel_weight_fraction),
    )
    pixel_weights = np.full(
        len(pixel_samples),
        supplemental_total / float(len(pixel_samples)),
        dtype=np.float64,
    )

    samples = np.vstack([region_lab, pixel_samples]).astype(np.float32)
    weights = np.concatenate([region_weights, pixel_weights]).astype(np.float64)
    return samples, weights, region_lab, flat_pixels


def _fit_compressed_initial_palette(
    img: np.ndarray,
    region_lab: np.ndarray,
    region_weights: np.ndarray,
    samples: np.ndarray,
    sample_weights: np.ndarray,
    *,
    initial_k: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the over-clustered palette and assign every source pixel."""
    model = _fit_kmeans(
        samples,
        initial_k,
        sample_weight=sample_weights,
        random_state=random_state,
    )

    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    cluster_map = _assign_pixels_to_centroids_lab(
        image_lab.reshape(-1, 3),
        model.cluster_centers_,
    ).reshape(img.shape[:2])

    cluster_colors = _lab_to_bgr_colors(model.cluster_centers_)

    # Primary importance still comes from region counts/capped areas.
    region_clusters = model.predict(region_lab)
    region_importance = np.bincount(
        region_clusters,
        weights=region_weights,
        minlength=initial_k,
    ).astype(np.float64)

    # Small pixel-support term prevents supplemental centroids from having
    # zero importance while keeping region weighting dominant.
    pixel_counts = np.bincount(
        cluster_map.reshape(-1),
        minlength=initial_k,
    ).astype(np.float64)
    if float(np.sum(pixel_counts)) > 0:
        pixel_counts /= float(np.sum(pixel_counts))
    pixel_support = pixel_counts * max(1.0, float(np.sum(region_weights)) * 0.05)

    cluster_weights = np.maximum(
        region_importance + pixel_support,
        1e-12,
    )
    return cluster_map, cluster_colors, cluster_weights


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def derive_palette(
    img: np.ndarray,
    palette_size: int = 32,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Original raw-pixel LAB K-means baseline."""
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    k = min(max(1, int(palette_size)), len(pixels))

    model = _fit_kmeans(pixels, k, random_state=random_state)
    return (
        model.labels_.reshape(img.shape[:2]),
        _lab_to_bgr_colors(model.cluster_centers_),
    )


def derive_palette_region_balanced(
    img: np.ndarray,
    palette_size: int = 32,
    edge_threshold: Threshold = 80,
    weight_mode: str = "uniform",
    area_cap: int = 2048,
    *,
    min_region_size: int = 24,
    segmentation_blur_ksize: int = 5,
    edge_dilate_iter: int = 1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Region-balanced K-means without compression."""
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    region_labels = segment_image(
        img,
        edge_threshold=edge_threshold,
        blur_ksize=segmentation_blur_ksize,
        edge_dilate_iter=edge_dilate_iter,
        min_region_size=min_region_size,
    )
    stats = compute_region_stats(img, region_labels)

    if not stats:
        color_map, colors = derive_palette(
            img,
            palette_size=1,
            random_state=random_state,
        )
        return color_map, colors, region_labels

    areas = np.asarray([stat.area for stat in stats], dtype=np.float64)
    weights = _region_weights(areas, weight_mode, area_cap)
    k = min(max(1, int(palette_size)), len(stats))

    _, cluster_map, colors, _ = _fit_region_palette(
        img,
        region_labels,
        stats,
        initial_k=k,
        sample_weights=weights,
        random_state=random_state,
    )
    return cluster_map, colors, region_labels


# ---------------------------------------------------------------------------
# Exact compressed palette
# ---------------------------------------------------------------------------


def derive_palette_compressed(
    img: np.ndarray,
    palette_size: int = 32,
    *,
    edge_threshold: Threshold = 80,
    weight_mode: str = "uniform",
    area_cap: int = 2048,
    min_region_size: int = 24,
    segmentation_blur_ksize: int = 5,
    edge_dilate_iter: int = 1,
    exclude_outline: bool = False,
    outline_color_bin_size: int = 16,
    outline_minimum_edge_pixels: int = 16,
    outline_minimum_support_fraction: float = 0.03,
    overcluster_factor: float = 2.0,
    gradient_tolerance: float = 1.0,
    solid_max_distance: float = 4.0,
    minimum_dither_span: float = 6.0,
    minimum_linearity: float = 0.90,
    maximum_reconstruction_rmse: float = 3.0,
    maximum_point_error: float = 6.0,
    minimum_endpoint_support_fraction: float = 0.05,
    endpoint_quantile: float = 0.05,
    forced_solid_penalty: float = 4.0,
    dither_penalty: float = 0.0,
    max_pixel_samples: int = 20000,
    pixel_fallback_weight_fraction: float = 0.10,
    random_state: int = 42,
) -> CompressedPaletteResult:
    """
    Build an exact-K region-balanced compressed palette.

    The initial K-means palette deliberately contains more than palette_size
    representatives. The exact-budget compressor then selects only the lowest
    error solid/dither compressions required to return exactly palette_size
    distinct output colors.

    gradient_tolerance is a user-facing multiplier for dither-family
    eligibility. 1.0 keeps the explicit thresholds; values above 1.0 relax
    linearity/error/span/support requirements, while values below 1.0 tighten
    them.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    requested_k = max(1, int(palette_size))
    overcluster_factor = max(1.0, float(overcluster_factor))

    region_labels = segment_image(
        img,
        edge_threshold=edge_threshold,
        blur_ksize=segmentation_blur_ksize,
        edge_dilate_iter=edge_dilate_iter,
        min_region_size=min_region_size,
    )
    stats = compute_region_stats(img, region_labels)

    outline_record = None
    if exclude_outline:
        outline_record = detect_outline_record(
            img,
            edge_threshold=edge_threshold,
            blur_ksize=segmentation_blur_ksize,
            color_bin_size=outline_color_bin_size,
            minimum_edge_pixels=outline_minimum_edge_pixels,
            minimum_support_fraction=(
                outline_minimum_support_fraction
            ),
        )

    if not stats:
        raise ValueError("Segmentation produced no usable regions.")

    areas = np.asarray([stat.area for stat in stats], dtype=np.float64)
    region_weights = _region_weights(areas, weight_mode, area_cap)

    desired_initial_k = max(
        requested_k,
        int(np.ceil(requested_k * overcluster_factor)),
    )

    samples, sample_weights, region_lab, _ = _hybrid_palette_samples(
        img,
        stats,
        region_weights,
        minimum_sample_count=desired_initial_k,
        max_pixel_samples=max_pixel_samples,
        pixel_weight_fraction=pixel_fallback_weight_fraction,
        random_state=random_state,
    )

    initial_k = min(len(samples), desired_initial_k)

    # Increase K when conversion to 8-bit BGR accidentally produces duplicate
    # representatives. Supplemental source pixels make this possible even when
    # watershed returns only a few large regions.
    while True:
        cluster_map, cluster_colors, cluster_weights = (
            _fit_compressed_initial_palette(
                img,
                region_lab,
                region_weights,
                samples,
                sample_weights,
                initial_k=initial_k,
                random_state=random_state,
            )
        )

        distinct_count = len(np.unique(cluster_colors, axis=0))
        if distinct_count >= requested_k:
            break

        if initial_k >= len(samples):
            raise ValueError(
                "Unable to obtain enough distinct initial representatives for "
                f"an exact {requested_k}-color palette. Distinct initial "
                f"colors={distinct_count}."
            )

        initial_k = min(
            len(samples),
            max(initial_k + 1, int(np.ceil(initial_k * 1.25))),
        )

    effective_thresholds = _resolve_gradient_tolerance(
        gradient_tolerance,
        minimum_dither_span=minimum_dither_span,
        minimum_linearity=minimum_linearity,
        maximum_reconstruction_rmse=maximum_reconstruction_rmse,
        maximum_point_error=maximum_point_error,
        minimum_endpoint_support_fraction=(
            minimum_endpoint_support_fraction
        ),
    )

    compression = compress_representatives(
        cluster_colors, # type: ignore
        weights=cluster_weights, # type: ignore
        target_size=requested_k,
        color_order="BGR",
        solid_max_distance=solid_max_distance,
        minimum_dither_span=(
            effective_thresholds["minimum_dither_span"]
        ),
        minimum_linearity=(
            effective_thresholds["minimum_linearity"]
        ),
        maximum_reconstruction_rmse=(
            effective_thresholds["maximum_reconstruction_rmse"]
        ),
        maximum_point_error=(
            effective_thresholds["maximum_point_error"]
        ),
        minimum_endpoint_support_fraction=(
            effective_thresholds["minimum_endpoint_support_fraction"]
        ),
        endpoint_quantile=endpoint_quantile,
        forced_solid_penalty=forced_solid_penalty,
        dither_penalty=dither_penalty,
    )

    return CompressedPaletteResult(
        cluster_map=cluster_map,
        cluster_colors=cluster_colors,
        cluster_weights=cluster_weights,
        compression=compression,
        region_labels=region_labels,
        outline_record=outline_record,
        requested_k=requested_k,
        initial_k=initial_k,
        compressed_k=compression.actual_size,
    )
