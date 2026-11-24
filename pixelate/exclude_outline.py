"""
Outline-color detection and grid-aware exclusion.
=================================================

This module addresses a downsampling artifact common in outlined drawings:

    dense source outlines
    -> one downsampled grid cell is dominated by the outline color
    -> the complete rendered cell becomes a black/dark square

The module has two stages:

1. detect_outline_record(...)
   Compute the Canny edge mask and estimate the most frequent quantized BGR
   color on those edge pixels.

2. replace_outline_grid_cells(...)
   Find rendered grid cells that both:
       - overlap source edge pixels; and
       - are dominated by the detected outline color.
   Replace those cells using nearby non-outline colors already present in the
   rendered image.

No new colors are invented. Fallback colors are selected from neighboring
rendered cells, so the output remains within the selected palette.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Union

import cv2
import numpy as np

from segment_image import compute_edge_mask as _segmentation_edge_mask


Threshold = Union[int, float, Tuple[Union[int, float], Union[int, float]]]


@dataclass(frozen=True)
class OutlineRecord:
    """Outline metadata recorded during the segmentation stage."""

    edge_mask: np.ndarray
    outline_color_bgr: np.ndarray | None
    edge_pixel_count: int
    outline_pixel_count: int
    support_fraction: float
    color_bin_size: int

    @property
    def valid(self) -> bool:
        return (
            self.outline_color_bgr is not None
            and self.edge_pixel_count > 0
            and self.outline_pixel_count > 0
        )


@dataclass(frozen=True)
class OutlineReplacementStats:
    """Diagnostics returned by the grid-aware replacement pass."""

    candidate_cells: int
    replaced_cells: int
    unresolved_cells: int


def _parse_edge_threshold(edge_threshold: Threshold) -> tuple[int, int]:
    if isinstance(edge_threshold, Iterable) and not isinstance(
        edge_threshold,
        (str, bytes),
    ):
        values = list(edge_threshold)
        if len(values) != 2:
            raise ValueError(
                "edge_threshold must be a scalar or a two-value pair."
            )
        low, high = int(values[0]), int(values[1])
    else:
        low = int(edge_threshold)
        high = int(round(float(edge_threshold) * 2.0))

    low = int(np.clip(low, 0, 255))
    high = int(np.clip(high, low + 1, 255))
    return low, high


def compute_edge_mask(
    image: np.ndarray,
    edge_threshold: Threshold = 80,
    *,
    blur_ksize: int = 5,
) -> np.ndarray:
    """Return the boolean form of the segmentation Canny edge mask."""
    return (
        _segmentation_edge_mask(
            image,
            edge_threshold,
            blur_ksize=blur_ksize,
        )
        > 0
    )


def _quantized_outline_color(
    edge_colors: np.ndarray,
    color_bin_size: int,
) -> tuple[np.ndarray, int]:
    """
    Return the modal quantized edge-color bin and its population.

    Exact source colors are often split by antialiasing or compression.
    Quantized counting therefore gives a more stable "most frequent color"
    estimate while retaining the user's intended mode-color rule.
    """
    bin_size = int(color_bin_size)
    if bin_size < 1 or bin_size > 256:
        raise ValueError("color_bin_size must be in [1, 256].")

    colors = np.asarray(edge_colors, dtype=np.uint8).reshape(-1, 3)
    quantized = colors.astype(np.int32) // bin_size

    bins, inverse, counts = np.unique(
        quantized,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    maximum = int(np.max(counts))
    tied = np.flatnonzero(counts == maximum)

    if len(tied) == 1:
        selected_bin = int(tied[0])
    else:
        # A deterministic tie-break biased toward the darker candidate is
        # useful for conventional outline art without overriding frequency.
        candidate_colors = []
        for bin_index in tied:
            members = colors[inverse == int(bin_index)]
            candidate_colors.append(np.median(members, axis=0))

        candidate_colors = np.asarray(candidate_colors, dtype=np.float32)
        luminance = (
            0.114 * candidate_colors[:, 0]
            + 0.587 * candidate_colors[:, 1]
            + 0.299 * candidate_colors[:, 2]
        )
        selected_bin = int(tied[int(np.argmin(luminance))])

    members = colors[inverse == selected_bin]
    representative = np.clip(
        np.rint(np.median(members.astype(np.float32), axis=0)),
        0,
        255,
    ).astype(np.uint8)

    return representative, int(len(members))


def detect_outline_record(
    image: np.ndarray,
    edge_threshold: Threshold = 80,
    *,
    blur_ksize: int = 5,
    color_bin_size: int = 16,
    minimum_edge_pixels: int = 16,
    minimum_support_fraction: float = 0.03,
    edge_sample_radius: int = 1,
    prefer_darker_side: bool = True,
) -> OutlineRecord:
    """
    Record the dominant color found on Canny edge pixels.

    If the edge population or modal-color support is too small, the returned
    record is invalid and no output replacement should be attempted.
    """
    edge_mask = compute_edge_mask(
        image,
        edge_threshold,
        blur_ksize=blur_ksize,
    )

    edge_pixel_count = int(np.count_nonzero(edge_mask))
    minimum_edge_pixels = max(1, int(minimum_edge_pixels))
    minimum_support_fraction = float(np.clip(
        minimum_support_fraction,
        0.0,
        1.0,
    ))

    if edge_pixel_count < minimum_edge_pixels:
        return OutlineRecord(
            edge_mask=edge_mask,
            outline_color_bgr=None,
            edge_pixel_count=edge_pixel_count,
            outline_pixel_count=0,
            support_fraction=0.0,
            color_bin_size=int(color_bin_size),
        )

    sample_colors = image[edge_mask]

    if prefer_darker_side and int(edge_sample_radius) > 0:
        # Canny marks the intensity transition rather than the centre of a
        # thick stroke. For each edge coordinate, sample the darkest pixel in
        # a small local neighborhood. This associates the edge with the dark
        # outline side instead of the bright fill/background side.
        radius = max(1, int(edge_sample_radius))
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        padded_image = cv2.copyMakeBorder(
            image,
            radius,
            radius,
            radius,
            radius,
            cv2.BORDER_REPLICATE,
        )
        padded_luminance = cv2.copyMakeBorder(
            luminance,
            radius,
            radius,
            radius,
            radius,
            cv2.BORDER_REPLICATE,
        )

        ys, xs = np.nonzero(edge_mask)
        sampled = np.empty((len(ys), 3), dtype=np.uint8)
        for index, (y, x) in enumerate(zip(ys, xs)):
            patch_l = padded_luminance[
                y:y + 2 * radius + 1,
                x:x + 2 * radius + 1,
            ]
            local_y, local_x = np.unravel_index(
                int(np.argmin(patch_l)),
                patch_l.shape,
            )
            sampled[index] = padded_image[
                y + local_y,
                x + local_x,
            ]
        sample_colors = sampled

    outline_color, outline_count = _quantized_outline_color(
        sample_colors,
        color_bin_size,
    )
    support = float(outline_count / max(edge_pixel_count, 1))

    if support < minimum_support_fraction:
        outline_color_or_none: np.ndarray | None = None
    else:
        outline_color_or_none = outline_color

    return OutlineRecord(
        edge_mask=edge_mask,
        outline_color_bgr=outline_color_or_none,
        edge_pixel_count=edge_pixel_count,
        outline_pixel_count=outline_count,
        support_fraction=support,
        color_bin_size=int(color_bin_size),
    )


def _bgr_to_standard_lab(colors_bgr: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors_bgr, dtype=np.float32)
    original_shape = colors.shape
    if original_shape[-1] != 3:
        raise ValueError("BGR colors must end with three channels.")

    normalized = np.clip(colors, 0.0, 255.0).reshape(1, -1, 3)
    normalized /= 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    return lab.reshape(original_shape).astype(np.float32)


def _outline_distance_map(
    output: np.ndarray,
    outline_color_bgr: np.ndarray,
) -> np.ndarray:
    output_lab = _bgr_to_standard_lab(output)
    outline_lab = _bgr_to_standard_lab(
        np.asarray(outline_color_bgr, dtype=np.uint8).reshape(1, 1, 3)
    )[0, 0]
    return np.linalg.norm(
        output_lab - outline_lab[None, None, :],
        axis=2,
    )


def _dominant_non_outline_color(
    cell: np.ndarray,
    outline_color_bgr: np.ndarray,
    color_tolerance: float,
) -> np.ndarray | None:
    """Return the most frequent non-outline color in one rendered cell."""
    pixels = np.asarray(cell, dtype=np.uint8).reshape(-1, 3)
    if len(pixels) == 0:
        return None

    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    distances = _outline_distance_map(
        unique.reshape(1, -1, 3),
        outline_color_bgr,
    ).reshape(-1)

    valid = distances > float(color_tolerance)
    if not np.any(valid):
        return None

    valid_indices = np.flatnonzero(valid)
    chosen = int(valid_indices[int(np.argmax(counts[valid_indices]))])
    return unique[chosen].astype(np.uint8)


def _cell_bounds(
    row: int,
    col: int,
    block_size: int,
) -> tuple[int, int, int, int]:
    y0 = row * block_size
    x0 = col * block_size
    return y0, y0 + block_size, x0, x0 + block_size


def replace_outline_grid_cells(
    output: np.ndarray,
    source_image: np.ndarray,
    outline_record: OutlineRecord | None,
    *,
    block_size: int,
    color_tolerance: float = 12.0,
    edge_coverage_threshold: float = 0.05,
    outline_fraction_threshold: float = 0.60,
    neighbor_radius: int = 2,
    spatial_penalty: float = 0.35,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, OutlineReplacementStats]:
    """
    Replace downsampled cells dominated by the recorded outline color.

    A cell is a candidate only when:
      1. enough source pixels in the cell are Canny edges; and
      2. enough rendered pixels in the cell are close to the outline color.

    Fallback colors are selected from nearby non-candidate cells. The selected
    color minimizes:

        LAB distance to the target source-block mean
        + spatial_penalty * grid distance

    The complete grid cell is replaced because the artifact is created at the
    grid/downsampling level rather than at individual source-pixel level.
    """
    if output is None or output.size == 0:
        raise ValueError("output is empty or None.")
    if source_image is None or source_image.size == 0:
        raise ValueError("source_image is empty or None.")

    if (
        outline_record is None
        or not outline_record.valid
        or outline_record.outline_color_bgr is None
    ):
        stats = OutlineReplacementStats(0, 0, 0)
        return (output.copy(), stats) if return_stats else output.copy()

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    height, width = output.shape[:2]
    if source_image.shape[0] < height or source_image.shape[1] < width:
        raise ValueError("source_image must cover the rendered output size.")
    if outline_record.edge_mask.shape[0] < height or outline_record.edge_mask.shape[1] < width:
        raise ValueError("outline edge mask must cover the rendered output size.")

    grid_height = height // block_size
    grid_width = width // block_size
    cropped_height = grid_height * block_size
    cropped_width = grid_width * block_size

    result = output[:cropped_height, :cropped_width].copy()
    source = source_image[:cropped_height, :cropped_width]
    edge_mask = outline_record.edge_mask[:cropped_height, :cropped_width]

    color_tolerance = max(0.0, float(color_tolerance))
    edge_coverage_threshold = float(np.clip(
        edge_coverage_threshold,
        0.0,
        1.0,
    ))
    outline_fraction_threshold = float(np.clip(
        outline_fraction_threshold,
        0.0,
        1.0,
    ))
    neighbor_radius = max(1, int(neighbor_radius))
    spatial_penalty = max(0.0, float(spatial_penalty))

    distances = _outline_distance_map(
        result,
        outline_record.outline_color_bgr,
    )
    close_to_outline = distances <= color_tolerance

    candidate_map = np.zeros(
        (grid_height, grid_width),
        dtype=bool,
    )
    edge_coverages = np.zeros_like(candidate_map, dtype=np.float32)
    outline_fractions = np.zeros_like(candidate_map, dtype=np.float32)

    for row in range(grid_height):
        for col in range(grid_width):
            y0, y1, x0, x1 = _cell_bounds(row, col, block_size)
            edge_coverage = float(np.mean(edge_mask[y0:y1, x0:x1]))
            outline_fraction = float(np.mean(
                close_to_outline[y0:y1, x0:x1]
            ))

            edge_coverages[row, col] = edge_coverage
            outline_fractions[row, col] = outline_fraction
            candidate_map[row, col] = (
                edge_coverage >= edge_coverage_threshold
                and outline_fraction >= outline_fraction_threshold
            )

    source_lab = _bgr_to_standard_lab(source)
    replaced = 0
    unresolved = 0

    # Read fallbacks from the original rendered image, not cells modified
    # earlier in this pass. This prevents a replacement color from cascading.
    original = result.copy()

    for row, col in np.argwhere(candidate_map):
        y0, y1, x0, x1 = _cell_bounds(
            int(row),
            int(col),
            block_size,
        )
        target_mean_lab = np.mean(
            source_lab[y0:y1, x0:x1],
            axis=(0, 1),
        )

        best_color: np.ndarray | None = None
        best_score = float("inf")

        for radius in range(1, neighbor_radius + 1):
            row_min = max(0, int(row) - radius)
            row_max = min(grid_height - 1, int(row) + radius)
            col_min = max(0, int(col) - radius)
            col_max = min(grid_width - 1, int(col) + radius)

            found_at_radius = False

            for neighbor_row in range(row_min, row_max + 1):
                for neighbor_col in range(col_min, col_max + 1):
                    if (
                        neighbor_row == int(row)
                        and neighbor_col == int(col)
                    ):
                        continue
                    if candidate_map[neighbor_row, neighbor_col]:
                        continue

                    ny0, ny1, nx0, nx1 = _cell_bounds(
                        neighbor_row,
                        neighbor_col,
                        block_size,
                    )
                    fallback = _dominant_non_outline_color(
                        original[ny0:ny1, nx0:nx1],
                        outline_record.outline_color_bgr,
                        color_tolerance,
                    )
                    if fallback is None:
                        continue

                    fallback_lab = _bgr_to_standard_lab(
                        fallback.reshape(1, 1, 3)
                    )[0, 0]
                    color_error = float(np.linalg.norm(
                        fallback_lab - target_mean_lab
                    ))
                    spatial_distance = max(
                        abs(neighbor_row - int(row)),
                        abs(neighbor_col - int(col)),
                    )
                    score = (
                        color_error
                        + spatial_penalty * float(spatial_distance)
                    )

                    if score < best_score:
                        best_score = score
                        best_color = fallback
                        found_at_radius = True

            if found_at_radius and best_color is not None:
                # Prefer the nearest radius containing a plausible fallback.
                break

        if best_color is None:
            unresolved += 1
            continue

        result[y0:y1, x0:x1] = best_color
        replaced += 1

    stats = OutlineReplacementStats(
        candidate_cells=int(np.count_nonzero(candidate_map)),
        replaced_cells=replaced,
        unresolved_cells=unresolved,
    )

    if return_stats:
        return result, stats
    return result


def replace_outline_pixels(
    output: np.ndarray,
    outline_record: OutlineRecord | None,
    *,
    color_tolerance: float = 12.0,
    neighbor_radius: int = 2,
) -> np.ndarray:
    """
    Literal per-pixel alternative.

    This follows the exact source-edge intersection rule but is less suitable
    for full-cell downsampling artifacts. The integrated pipeline uses
    replace_outline_grid_cells instead.
    """
    if (
        outline_record is None
        or not outline_record.valid
        or outline_record.outline_color_bgr is None
    ):
        return output.copy()

    result = output.copy()
    height, width = result.shape[:2]
    edge_mask = outline_record.edge_mask[:height, :width]
    distance = _outline_distance_map(
        result,
        outline_record.outline_color_bgr,
    )
    candidate = edge_mask & (distance <= float(color_tolerance))
    source = result.copy()

    for y, x in np.argwhere(candidate):
        selected = None
        for radius in range(1, max(1, int(neighbor_radius)) + 1):
            y0 = max(0, int(y) - radius)
            y1 = min(height, int(y) + radius + 1)
            x0 = max(0, int(x) - radius)
            x1 = min(width, int(x) + radius + 1)

            patch = source[y0:y1, x0:x1]
            patch_distance = _outline_distance_map(
                patch,
                outline_record.outline_color_bgr,
            )
            valid = patch_distance > float(color_tolerance)
            if np.any(valid):
                colors, counts = np.unique(
                    patch[valid].reshape(-1, 3),
                    axis=0,
                    return_counts=True,
                )
                selected = colors[int(np.argmax(counts))]
                break

        if selected is not None:
            result[int(y), int(x)] = selected

    return result
