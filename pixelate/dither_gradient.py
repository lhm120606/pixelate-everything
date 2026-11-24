"""
Endpoint dithering for gradient regions in MVP ver.4.
=====================================================

Purpose
-------
Represent a gradient region with two endpoint palette colors, then simulate
intermediate tones by spatial dithering instead of allocating many palette
representatives to every shade in the gradient.

All public functions use OpenCV's BGR color order by default, consistent with
this project. Use color_order="RGB" if you want RGB inputs/outputs.

Core API
--------
generate_dithered_colors(color_a, color_b, n_between, ...)
    Given two endpoint colors and a requested number of intermediate visual
    steps, return dithered patches for the endpoints and intermediate ratios.

dither_two_colors(color_a, color_b, ratio, size, ...)
    Build one dithered patch that visually approximates:
        (1-ratio) * color_a + ratio * color_b
    but uses only color_a and color_b as actual output pixels.

save_dither_preview(levels, out_path)
    Save a horizontal debug strip for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, Tuple

import cv2
import numpy as np

ColorOrder = Literal["BGR", "RGB"]
DitherMethod = Literal["ordered", "floyd_steinberg"]


@dataclass(frozen=True)
class DitheredColor:
    """One visual tone in a two-endpoint dither ramp."""

    index: int
    ratio: float
    # Solid interpolated reference color. This is for analysis/preview only.
    # It should not be treated as a new palette representative if the goal is
    # endpoint-only gradient preservation.
    reference_color: np.ndarray
    # Actual dithered patch. Pixels contain only endpoint colors.
    patch: np.ndarray


def _as_color(color: Sequence[int | float], color_order: ColorOrder) -> np.ndarray:
    """Convert a 3-channel color to uint8 BGR."""
    arr = np.asarray(color, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError("color must contain exactly 3 channels.")
    arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    if color_order == "RGB":
        arr = arr[::-1]
    elif color_order != "BGR":
        raise ValueError('color_order must be "BGR" or "RGB".')
    return arr


def _from_bgr(color_bgr: np.ndarray, color_order: ColorOrder) -> np.ndarray:
    """Convert internal BGR color to requested output order."""
    if color_order == "RGB":
        return color_bgr[::-1].copy()
    return color_bgr.copy()


def bayer_matrix(size: int = 8) -> np.ndarray:
    """
    Return a normalized Bayer threshold matrix in [0, 1).

    size must be a power of two. 8 is a good default for pixel-art-like ordered
    dithering because it creates a stable pattern without excessive repetition.
    """
    size = int(size)
    if size < 2 or (size & (size - 1)) != 0:
        raise ValueError("Bayer matrix size must be a power of two >= 2.")

    mat = np.array([[0, 2], [3, 1]], dtype=np.float64)
    n = 2
    while n < size:
        mat = np.block([
            [4 * mat + 0, 4 * mat + 2],
            [4 * mat + 3, 4 * mat + 1],
        ])
        n *= 2
    return (mat + 0.5) / float(size * size)


def interpolate_color(
    color_a: Sequence[int | float],
    color_b: Sequence[int | float],
    ratio: float,
    *,
    color_space: Literal["lab", "bgr"] = "lab",
    color_order: ColorOrder = "BGR",
) -> np.ndarray:
    """
    Return the solid reference color for a given ratio.

    This is useful for debugging and for comparing the dithered visual tone to
    a continuous gradient. It is not needed by the endpoint-only palette itself.
    """
    ratio = float(np.clip(ratio, 0.0, 1.0))
    a = _as_color(color_a, color_order).astype(np.float32)
    b = _as_color(color_b, color_order).astype(np.float32)

    if color_space == "bgr":
        out_bgr = (1.0 - ratio) * a + ratio * b
    elif color_space == "lab":
        pair = np.array([[a, b]], dtype=np.uint8)
        pair_lab = cv2.cvtColor(pair, cv2.COLOR_BGR2LAB).reshape(2, 3).astype(np.float32)
        out_lab = (1.0 - ratio) * pair_lab[0] + ratio * pair_lab[1]
        out_bgr = cv2.cvtColor(out_lab.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).reshape(3)
    else:
        raise ValueError('color_space must be "lab" or "bgr".')

    out_bgr = np.clip(np.rint(out_bgr), 0, 255).astype(np.uint8)
    return _from_bgr(out_bgr, color_order)


def _ordered_mask(ratio: float, size: Tuple[int, int], matrix_size: int) -> np.ndarray:
    """Binary mask where True pixels use color_b."""
    h, w = int(size[0]), int(size[1])
    if h <= 0 or w <= 0:
        raise ValueError("size must be (height, width) with positive values.")
    thresholds = bayer_matrix(matrix_size)
    tiled = np.tile(
        thresholds,
        (int(np.ceil(h / matrix_size)), int(np.ceil(w / matrix_size))),
    )[:h, :w]
    return tiled < float(np.clip(ratio, 0.0, 1.0))


def _floyd_steinberg_mask(ratio: float, size: Tuple[int, int]) -> np.ndarray:
    """
    Binary error-diffusion mask for a constant target ratio.

    For a gradient region, call this per local ratio or use ordered dithering
    over a ratio map. For this small endpoint module, a constant ratio patch is
    enough to generate visual intermediate tones.
    """
    h, w = int(size[0]), int(size[1])
    if h <= 0 or w <= 0:
        raise ValueError("size must be (height, width) with positive values.")

    arr = np.full((h, w), float(np.clip(ratio, 0.0, 1.0)), dtype=np.float64)
    out = np.zeros((h, w), dtype=bool)

    for y in range(h):
        # Serpentine traversal reduces directional artifacts.
        if y % 2 == 0:
            xs = range(w)
            direction = 1
        else:
            xs = range(w - 1, -1, -1)
            direction = -1

        for x in xs:
            old = arr[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new

            x1 = x + direction
            x_1 = x - direction
            if 0 <= x1 < w:
                arr[y, x1] += err * 7.0 / 16.0
            if y + 1 < h:
                if 0 <= x_1 < w:
                    arr[y + 1, x_1] += err * 3.0 / 16.0
                arr[y + 1, x] += err * 5.0 / 16.0
                if 0 <= x1 < w:
                    arr[y + 1, x1] += err * 1.0 / 16.0

    return out


def dither_two_colors(
    color_a: Sequence[int | float],
    color_b: Sequence[int | float],
    ratio: float,
    *,
    size: Tuple[int, int] = (32, 32),
    method: DitherMethod = "ordered",
    matrix_size: int = 8,
    color_order: ColorOrder = "BGR",
) -> np.ndarray:
    """
    Generate one dithered patch using only two endpoint colors.

    Parameters
    ----------
    color_a, color_b:
        Endpoint colors. BGR by default.
    ratio:
        Desired contribution of color_b, in [0, 1].
        0.0 -> all color_a, 1.0 -> all color_b.
    size:
        Output patch size as (height, width).
    method:
        "ordered" for deterministic Bayer dithering, or "floyd_steinberg" for
        error diffusion. Ordered is usually cleaner for pixel-art assets.
    matrix_size:
        Bayer matrix size for ordered dithering. Must be a power of two >= 2.
    color_order:
        "BGR" for OpenCV pipeline compatibility, or "RGB".

    Returns
    -------
    np.ndarray
        Patch image, shape (height, width, 3), uint8. Output color order matches
        color_order.
    """
    a_bgr = _as_color(color_a, color_order)
    b_bgr = _as_color(color_b, color_order)

    ratio = float(np.clip(ratio, 0.0, 1.0))
    if method == "ordered":
        mask = _ordered_mask(ratio, size, matrix_size)
    elif method == "floyd_steinberg":
        mask = _floyd_steinberg_mask(ratio, size)
    else:
        raise ValueError('method must be "ordered" or "floyd_steinberg".')

    h, w = int(size[0]), int(size[1])
    patch_bgr = np.empty((h, w, 3), dtype=np.uint8)
    patch_bgr[~mask] = a_bgr
    patch_bgr[mask] = b_bgr

    if color_order == "RGB":
        return patch_bgr[:, :, ::-1].copy()
    return patch_bgr



def _ordered_mask_from_ratio_map(
    ratio_map: np.ndarray,
    matrix_size: int,
    *,
    origin: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    Convert a per-pixel 0..1 ratio map into a Bayer dither mask.

    origin is (y, x) in the final image. Keeping a global origin prevents the
    Bayer pattern from restarting at every grid cell.
    """
    ratios = np.clip(np.asarray(ratio_map, dtype=np.float32), 0.0, 1.0)
    if ratios.ndim != 2:
        raise ValueError("ratio_map must be a 2-D array.")

    thresholds = bayer_matrix(matrix_size)
    matrix_height, matrix_width = thresholds.shape

    y0, x0 = int(origin[0]), int(origin[1])
    y_index = (np.arange(ratios.shape[0]) + y0) % matrix_height
    x_index = (np.arange(ratios.shape[1]) + x0) % matrix_width
    threshold_map = thresholds[np.ix_(y_index, x_index)]

    return ratios > threshold_map


def _floyd_steinberg_mask_from_ratio_map(
    ratio_map: np.ndarray,
) -> np.ndarray:
    """Error-diffuse a spatially varying 0..1 ratio map."""
    arr = np.clip(
        np.asarray(ratio_map, dtype=np.float64),
        0.0,
        1.0,
    ).copy()

    if arr.ndim != 2:
        raise ValueError("ratio_map must be a 2-D array.")

    height, width = arr.shape
    output = np.zeros((height, width), dtype=bool)

    for y in range(height):
        if y % 2 == 0:
            x_values = range(width)
            direction = 1
        else:
            x_values = range(width - 1, -1, -1)
            direction = -1

        for x in x_values:
            old_value = arr[y, x]
            new_value = 1.0 if old_value >= 0.5 else 0.0
            output[y, x] = bool(new_value)
            error = old_value - new_value

            next_x = x + direction
            previous_x = x - direction

            if 0 <= next_x < width:
                arr[y, next_x] += error * 7.0 / 16.0

            if y + 1 < height:
                if 0 <= previous_x < width:
                    arr[y + 1, previous_x] += error * 3.0 / 16.0

                arr[y + 1, x] += error * 5.0 / 16.0

                if 0 <= next_x < width:
                    arr[y + 1, next_x] += error * 1.0 / 16.0

    return output


def dither_ratio_map(
    color_a: Sequence[int | float],
    color_b: Sequence[int | float],
    ratio_map: np.ndarray,
    *,
    method: DitherMethod = "ordered",
    matrix_size: int = 8,
    color_order: ColorOrder = "BGR",
    origin: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    Dither a spatial ratio map using exactly one endpoint pair.

    This function is intended to be called separately for every detected
    gradient region. The caller supplies that region's own endpoint colors.

    Parameters
    ----------
    color_a, color_b:
        Endpoint colors for one gradient region.
    ratio_map:
        Two-dimensional array. Each element gives the desired contribution of
        color_b at that location.
    origin:
        Absolute (y, x) coordinate of the patch in the final output. Used by
        ordered dithering to keep the Bayer phase continuous across grid cells.
    """
    ratios = np.clip(
        np.asarray(ratio_map, dtype=np.float32),
        0.0,
        1.0,
    )

    if ratios.ndim != 2:
        raise ValueError("ratio_map must be a 2-D array.")

    color_a_bgr = _as_color(color_a, color_order)
    color_b_bgr = _as_color(color_b, color_order)

    if method == "ordered":
        use_color_b = _ordered_mask_from_ratio_map(
            ratios,
            matrix_size,
            origin=origin,
        )
    elif method == "floyd_steinberg":
        use_color_b = _floyd_steinberg_mask_from_ratio_map(ratios)
    else:
        raise ValueError(
            'method must be "ordered" or "floyd_steinberg".'
        )

    patch_bgr = np.empty(
        (ratios.shape[0], ratios.shape[1], 3),
        dtype=np.uint8,
    )
    patch_bgr[~use_color_b] = color_a_bgr
    patch_bgr[use_color_b] = color_b_bgr

    if color_order == "RGB":
        return patch_bgr[:, :, ::-1].copy()

    return patch_bgr

def generate_dithered_colors(
    color_a: Sequence[int | float],
    color_b: Sequence[int | float],
    n_between: int,
    *,
    patch_size: int | Tuple[int, int] = 32,
    method: DitherMethod = "ordered",
    matrix_size: int = 8,
    include_endpoints: bool = True,
    color_space: Literal["lab", "bgr"] = "lab",
    color_order: ColorOrder = "BGR",
) -> list[DitheredColor]:
    """
    Generate a dither ramp between two colors.

    Example
    -------
    generate_dithered_colors([20, 40, 220], [230, 180, 60], 3)
    gives ratios:
        0.00, 0.25, 0.50, 0.75, 1.00     if include_endpoints=True
        0.25, 0.50, 0.75                 if include_endpoints=False

    The intermediate visual tones still use only the two endpoint colors in
    the actual patch pixels. The reference_color field is only a continuous
    comparison color.
    """
    n_between = int(n_between)
    if n_between < 0:
        raise ValueError("n_between must be >= 0.")

    if isinstance(patch_size, tuple):
        size = (int(patch_size[0]), int(patch_size[1]))
    else:
        size = (int(patch_size), int(patch_size))

    if include_endpoints:
        ratios = np.linspace(0.0, 1.0, n_between + 2)
    else:
        ratios = np.linspace(0.0, 1.0, n_between + 2)[1:-1]

    levels: list[DitheredColor] = []
    for i, ratio in enumerate(ratios):
        ref = interpolate_color(
            color_a,
            color_b,
            float(ratio),
            color_space=color_space,
            color_order=color_order,
        )
        patch = dither_two_colors(
            color_a,
            color_b,
            float(ratio),
            size=size,
            method=method,
            matrix_size=matrix_size,
            color_order=color_order,
        )
        levels.append(DitheredColor(index=i, ratio=float(ratio), reference_color=ref, patch=patch))
    return levels


def save_dither_preview(
    levels: Sequence[DitheredColor],
    out_path: str,
    *,
    gap: int = 4,
    color_order: ColorOrder = "BGR",
) -> np.ndarray:
    """
    Save a horizontal preview strip of generated dithered tones.

    Returns the preview image array in the requested color order.
    """
    if not levels:
        raise ValueError("levels must not be empty.")

    patches = [level.patch for level in levels]
    h = max(p.shape[0] for p in patches)
    w = sum(p.shape[1] for p in patches) + gap * (len(patches) - 1)
    preview = np.full((h, w, 3), 255, dtype=np.uint8)

    x = 0
    for p in patches:
        ph, pw = p.shape[:2]
        preview[:ph, x:x + pw] = p
        x += pw + gap

    to_write = preview[:, :, ::-1] if color_order == "RGB" else preview
    cv2.imwrite(out_path, to_write)
    return preview


if __name__ == "__main__":
    import argparse

    def parse_color(text: str) -> tuple[int, int, int]:
        parts = tuple(int(x.strip()) for x in text.split(","))
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("color must be formatted as c1,c2,c3")
        return parts

    parser = argparse.ArgumentParser(description="Generate endpoint-dithered gradient tones.")
    parser.add_argument("--color_a", type=parse_color, required=True,
                        help="First endpoint color, e.g. 30,60,220. BGR by default.")
    parser.add_argument("--color_b", type=parse_color, required=True,
                        help="Second endpoint color, e.g. 220,180,60. BGR by default.")
    parser.add_argument("--n_between", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--method", choices=["ordered", "floyd_steinberg"], default="ordered")
    parser.add_argument("--matrix_size", type=int, default=8)
    parser.add_argument("--color_order", choices=["BGR", "RGB"], default="BGR")
    parser.add_argument("--out", default="dither_preview.png")
    args = parser.parse_args()

    ramp = generate_dithered_colors(
        args.color_a,
        args.color_b,
        args.n_between,
        patch_size=args.patch_size,
        method=args.method,
        matrix_size=args.matrix_size,
        color_order=args.color_order,
    )
    save_dither_preview(ramp, args.out, color_order=args.color_order)
    print("Generated ratios:", ", ".join(f"{x.ratio:.3f}" for x in ramp))
    print(f"Saved preview -> {args.out}")
