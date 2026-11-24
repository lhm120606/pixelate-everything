"""
Grid rendering for MVP ver.3.
============================

The active compressed path uses palette families, not gradient-region labels.

For each output grid cell:

1. inspect the initial K-means cluster labels inside the cell;
2. select the dominant compressed family;
3. render one solid color, or dither that family's two endpoints inside the
   complete grid cell.

Segmentation and watershed labels are not used during rendering.
"""

from __future__ import annotations

import cv2
import numpy as np

from dither_gradient import dither_ratio_map
from exclude_outline import replace_outline_grid_cells
from palette import CompressedPaletteResult


def derive_block_color(
    block: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """Return the palette color nearest to the block's average BGR color."""
    average = np.average(block, axis=(0, 1))
    average = np.clip(average, 0, 255).astype(np.float32)

    palette = np.asarray(colors, dtype=np.float32)
    distances = np.linalg.norm(palette - average, axis=1)
    color_index = int(np.argmin(distances))
    return np.asarray(colors[color_index], dtype=np.uint8)


def pixelate_image(
    img: np.ndarray,
    block_size: int,
    color_map: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """Original low-resolution grid renderer."""
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    height, width = img.shape[:2]
    grid_width = width // block_size
    grid_height = height // block_size

    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("block_size is too large for the input image.")

    cropped_width = grid_width * block_size
    cropped_height = grid_height * block_size
    cropped = img[:cropped_height, :cropped_width]

    pixelated = np.zeros(
        (grid_height, grid_width, 3),
        dtype=np.uint8,
    )

    for row in range(grid_height):
        y0 = row * block_size
        y1 = y0 + block_size

        for col in range(grid_width):
            x0 = col * block_size
            x1 = x0 + block_size
            block = cropped[y0:y1, x0:x1]
            pixelated[row, col] = derive_block_color(
                block,
                colors,
            )

    return pixelated


def expand_pixel_grid(
    pixel_grid: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """Expand every low-resolution grid pixel into one solid block."""
    if pixel_grid is None or pixel_grid.size == 0:
        raise ValueError("pixel_grid is empty or None.")

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    height, width = pixel_grid.shape[:2]
    return cv2.resize(
        pixel_grid,
        (width * block_size, height * block_size),
        interpolation=cv2.INTER_NEAREST,
    )



def scale_grid_cells_to_pixels(
    image: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    Reduce every rendered grid cell to exactly one output pixel.

    The operation is an exact non-overlapping area average:

        (grid_h * block_size, grid_w * block_size, 3)
        -> (grid_h, grid_w, 3)

    Unlike nearest-neighbor resizing, this blends all colors inside each grid
    cell. This is important for dithered cells: their endpoint pattern becomes
    one representative blended color.

    Unlike a generic resize, cells never sample from neighboring cells because
    the image is reshaped into explicit block windows before averaging.
    """
    if image is None or image.size == 0:
        raise ValueError("image is empty or None.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3).")

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    height, width = image.shape[:2]
    grid_height = height // block_size
    grid_width = width // block_size

    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("block_size is larger than the rendered image.")

    cropped_height = grid_height * block_size
    cropped_width = grid_width * block_size
    cropped = image[:cropped_height, :cropped_width].astype(np.float32)

    # Explicit non-overlapping block average. Axis layout:
    # grid row, pixel row inside cell, grid col, pixel col inside cell, channel
    blocks = cropped.reshape(
        grid_height,
        block_size,
        grid_width,
        block_size,
        3,
    )
    reduced = blocks.mean(axis=(1, 3))

    return np.clip(
        np.rint(reduced),
        0,
        255,
    ).astype(np.uint8)


def finalize_grid_output(
    image: np.ndarray,
    block_size: int,
    *,
    preserve_size: bool = False,
) -> np.ndarray:
    """
    Return either the full rendered image or one pixel per grid cell.

    preserve_size=False is the Ver. 4 delivery default.
    preserve_size=True retains the full-resolution dither pattern.
    """
    if preserve_size:
        return np.asarray(image, dtype=np.uint8).copy()

    return scale_grid_cells_to_pixels(
        image,
        block_size,
    )



def _dominant_group(
    cluster_block: np.ndarray,
    palette_result: CompressedPaletteResult,
) -> tuple[int, np.ndarray]:
    """
    Return the dominant compressed family and its pixel mask in one grid cell.

    Several initial K-means clusters may belong to one compressed family. Their
    counts are therefore combined before choosing the dominant family.
    """
    cluster_ids = cluster_block.astype(np.int32)
    group_ids = palette_result.compression.group_index_by_color[cluster_ids]

    unique_groups, counts = np.unique(
        group_ids,
        return_counts=True,
    )
    dominant_index = int(np.argmax(counts))
    group_id = int(unique_groups[dominant_index])
    group_mask = group_ids == group_id
    return group_id, group_mask


def _mean_family_ratio(
    cluster_block: np.ndarray,
    group_mask: np.ndarray,
    palette_result: CompressedPaletteResult,
) -> float:
    """
    Average member ratios for the selected family using source-pixel counts.

    This keeps the renderer palette-only: the ratio comes from the initial
    K-means representative assigned to each source pixel, not from a spatial
    gradient detector.
    """
    cluster_ids = cluster_block[group_mask].astype(np.int32)
    ratios = palette_result.compression.ratio_by_color[cluster_ids]
    ratios = ratios[np.isfinite(ratios)]

    if len(ratios) == 0:
        return 0.5

    return float(np.clip(np.mean(ratios), 0.0, 1.0))



def _palette_entries_with_groups(
    palette_result: CompressedPaletteResult,
) -> list[tuple[np.ndarray, int]]:
    """Return every output palette color together with its group index."""
    entries: list[tuple[np.ndarray, int]] = []

    for group_index, group in enumerate(palette_result.compression.groups):
        if group.mode == "solid":
            entries.append((
                np.asarray(group.representative, dtype=np.uint8).reshape(3),
                group_index,
            ))
        else:
            entries.append((
                np.asarray(group.endpoint_a, dtype=np.uint8).reshape(3),
                group_index,
            ))
            entries.append((
                np.asarray(group.endpoint_b, dtype=np.uint8).reshape(3),
                group_index,
            ))

    return entries


def _ensure_every_palette_color_is_used(
    output: np.ndarray,
    source: np.ndarray,
    cluster_map: np.ndarray,
    palette_result: CompressedPaletteResult,
) -> np.ndarray:
    """
    Guarantee that every selected palette entry appears in the rendered image.

    Normally all entries are used naturally. If block aggregation suppresses a
    small family or one dither endpoint, this function inserts the missing
    color at the least-disruptive source pixel belonging to that family. It
    never introduces a color outside the selected palette.
    """
    result = output.copy()
    height, width = result.shape[:2]

    palette_entries = _palette_entries_with_groups(palette_result)
    palette = np.vstack([entry[0] for entry in palette_entries]).astype(np.uint8)

    if height * width < len(palette):
        raise ValueError(
            "The rendered image contains fewer pixels than palette entries, so "
            "all palette colors cannot be used."
        )

    flat = result.reshape(-1, 3)
    source_flat = source[:height, :width].reshape(-1, 3).astype(np.float32)

    cluster_flat = cluster_map[:height, :width].reshape(-1).astype(np.int32)
    group_flat = palette_result.compression.group_index_by_color[cluster_flat]

    def color_counts() -> dict[tuple[int, int, int], int]:
        unique, counts = np.unique(flat, axis=0, return_counts=True)
        return {
            tuple(int(x) for x in color): int(count)# type: ignore
            for color, count in zip(unique, counts)
        }

    counts = color_counts()

    for color, group_index in palette_entries:
        key = tuple(int(x) for x in color)
        if counts.get(key, 0) > 0: # type: ignore
            continue

        family_candidates = np.flatnonzero(group_flat == group_index)
        candidates = family_candidates
        if len(candidates) == 0:
            candidates = np.arange(len(flat), dtype=np.int64)

        # Avoid deleting the only occurrence of another palette color.
        replaceable = []
        for candidate in candidates:
            current_key = tuple(int(x) for x in flat[int(candidate)])
            if counts.get(current_key, 0) > 1: # type: ignore
                replaceable.append(int(candidate))

        if not replaceable:
            replaceable = [
                index
                for index in range(len(flat))
                if counts.get(
                    tuple(int(x) for x in flat[index]), # type: ignore
                    0,
                ) > 1 # type: ignore
            ]

        if not replaceable:
            raise RuntimeError(
                "Unable to place every palette color without removing another "
                "color's only occurrence."
            )

        replaceable_array = np.asarray(replaceable, dtype=np.int64)
        differences = (
            source_flat[replaceable_array]
            - color.astype(np.float32)[None, :]
        )
        best_local = int(np.argmin(np.sum(differences * differences, axis=1)))
        best_index = int(replaceable_array[best_local])

        old_key = tuple(int(x) for x in flat[best_index])
        flat[best_index] = color
        counts[old_key] -= 1 # type: ignore
        counts[key] = counts.get(key, 0) + 1 # type: ignore

    final_unique = len(np.unique(flat, axis=0))
    if final_unique != len(palette):
        raise RuntimeError(
            "Internal error: rendered output does not use the complete exact "
            f"palette ({final_unique} used, {len(palette)} selected)."
        )

    return result


def pixelate_image_with_compressed_palette(
    img: np.ndarray,
    block_size: int,
    palette_result: CompressedPaletteResult,
    *,
    dither_method: str = "ordered",
    matrix_size: int = 8,
    extreme_ratio_threshold: float = 0.08,
    ensure_palette_usage: bool = True,
    apply_outline_exclusion: bool = True,
    outline_color_tolerance: float = 12.0,
    outline_edge_coverage_threshold: float = 0.05,
    outline_fraction_threshold: float = 0.60,
    outline_neighbor_radius: int = 2,
    outline_spatial_penalty: float = 0.35,
) -> np.ndarray:
    """
    Render a compressed palette with dithering inside each output grid cell.

    Solid family:
        fill the complete cell with its one representative color.

    Dither family:
        average the ratios of the initial K-means clusters from that family in
        the cell, then mix the two family endpoints inside the cell.

    If a cell contains several families, the dominant family is selected. This
    matches the original one-representative-per-grid-cell pixelation behavior.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    if palette_result.cluster_map.shape[:2] != img.shape[:2]:
        raise ValueError(
            "palette_result.cluster_map must match the input image size."
        )

    height, width = img.shape[:2]
    grid_width = width // block_size
    grid_height = height // block_size

    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("block_size is too large for the input image.")

    cropped_width = grid_width * block_size
    cropped_height = grid_height * block_size
    cluster_map = palette_result.cluster_map[
        :cropped_height,
        :cropped_width,
    ]

    output = np.zeros(
        (cropped_height, cropped_width, 3),
        dtype=np.uint8,
    )

    threshold = float(np.clip(extreme_ratio_threshold, 0.0, 0.5))

    for row in range(grid_height):
        y0 = row * block_size
        y1 = y0 + block_size

        for col in range(grid_width):
            x0 = col * block_size
            x1 = x0 + block_size

            cluster_block = cluster_map[y0:y1, x0:x1]
            group_id, group_mask = _dominant_group(
                cluster_block,
                palette_result,
            )
            group = palette_result.compression.groups[group_id]

            if group.mode == "solid":
                output[y0:y1, x0:x1] = np.asarray(
                    group.representative,
                    dtype=np.uint8,
                )
                continue

            ratio = _mean_family_ratio(
                cluster_block,
                group_mask,
                palette_result,
            )

            endpoint_a = np.asarray(
                group.endpoint_a,
                dtype=np.uint8,
            )
            endpoint_b = np.asarray(
                group.endpoint_b,
                dtype=np.uint8,
            )

            # Avoid adding visible texture when the result is already almost
            # identical to one endpoint.
            if ratio <= threshold:
                output[y0:y1, x0:x1] = endpoint_a
                continue

            if ratio >= 1.0 - threshold:
                output[y0:y1, x0:x1] = endpoint_b
                continue

            ratio_map = np.full(
                (block_size, block_size),
                ratio,
                dtype=np.float32,
            )

            output[y0:y1, x0:x1] = dither_ratio_map(
                endpoint_a, # type: ignore
                endpoint_b, # type: ignore
                ratio_map,
                method=dither_method, # type: ignore
                matrix_size=matrix_size,
                color_order="BGR",
                origin=(y0, x0),
            )

    if ensure_palette_usage:
        output = _ensure_every_palette_color_is_used(
            output,
            img,
            cluster_map,
            palette_result,
        )

    if (
        apply_outline_exclusion
        and palette_result.outline_record is not None
    ):
        output = replace_outline_grid_cells(
            output,
            img,
            palette_result.outline_record,
            block_size=block_size,
            color_tolerance=outline_color_tolerance,
            edge_coverage_threshold=(
                outline_edge_coverage_threshold
            ),
            outline_fraction_threshold=(
                outline_fraction_threshold
            ),
            neighbor_radius=outline_neighbor_radius,
            spatial_penalty=outline_spatial_penalty,
        )

    return output # type: ignore
