"""
Small batch runner for the MVP ver.4 pipeline.
=====================================================

Modes:
    raw         raw-pixel LAB K-means
    region      watershed region-balanced K-means
    compressed  region-balanced K-means + recursive palette compression
    all         run all three modes for comparison

The old gradient-exclusion path is not used.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np

from derive_grid import expand_pixel_grid
from derive_grid import finalize_grid_output
from derive_grid import pixelate_image
from derive_grid import pixelate_image_with_compressed_palette
from palette import derive_palette
from palette import derive_palette_compressed
from palette import derive_palette_region_balanced
from smart_grid import estimate_block_size



IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


LOG_IDENTITY_COLUMNS = [
    "image_name",
    "output_name",
    "status",
    "error",
]

LOG_TOP_LEVEL_PARAMETER_COLUMNS = [
    "mode",
    "detail_mode",
    "weight_mode",
    "dither_method",
    "palette_size",
    "requested_block_size",
    "grid_source",
    "preserve_size",
    "exclude_outline",
    "palette_usage_enforcement",
]

LOG_GRID_PARAMETER_COLUMNS = [
    "grid_candidates",
]

LOG_SEGMENTATION_PARAMETER_COLUMNS = [
    "edge_threshold",
    "min_region_size",
    "segmentation_blur_ksize",
    "edge_dilate_iter",
    "area_cap",
]

LOG_COMPRESSION_PARAMETER_COLUMNS = [
    "overcluster_factor",
    "gradient_tolerance",
    "solid_max_distance",
    "minimum_dither_span",
    "minimum_linearity",
    "maximum_reconstruction_rmse",
    "maximum_point_error",
    "minimum_endpoint_support_fraction",
    "endpoint_quantile",
    "forced_solid_penalty",
    "dither_penalty",
]

LOG_DITHER_PARAMETER_COLUMNS = [
    "matrix_size",
    "extreme_ratio_threshold",
]

LOG_OUTLINE_PARAMETER_COLUMNS = [
    "outline_color_bin_size",
    "outline_minimum_edge_pixels",
    "outline_minimum_support_fraction",
    "outline_color_tolerance",
    "outline_edge_coverage_threshold",
    "outline_fraction_threshold",
    "outline_neighbor_radius",
    "outline_spatial_penalty",
]

LOG_GENERAL_PARAMETER_COLUMNS = [
    "random_state",
    "recursive",
    "overwrite",
    "verbose",
    "suffix",
    "output_format",
    "no_log",
    "log_name_override",
]

LOG_RUNTIME_COLUMNS = [
    "input_width",
    "input_height",
    "output_width",
    "output_height",
    "block_size",
    "requested_k",
    "initial_k",
    "compressed_k",
    "solid_groups",
    "dither_groups",
    "unique_palette_k",
    "forced_solid_groups",
    "segments",
    "elapsed_sec",
]

LOG_FIELDNAMES = (
    LOG_IDENTITY_COLUMNS
    + LOG_TOP_LEVEL_PARAMETER_COLUMNS
    + LOG_GRID_PARAMETER_COLUMNS
    + LOG_SEGMENTATION_PARAMETER_COLUMNS
    + LOG_COMPRESSION_PARAMETER_COLUMNS
    + LOG_DITHER_PARAMETER_COLUMNS
    + LOG_OUTLINE_PARAMETER_COLUMNS
    + LOG_GENERAL_PARAMETER_COLUMNS
    + LOG_RUNTIME_COLUMNS
)


def _format_slug_value(value: object) -> str:
    """Return a compact filesystem-safe value for automatic log names."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if value is None:
        return "auto"
    if isinstance(value, float):
        rendered = f"{value:.4g}"
    else:
        rendered = str(value)

    rendered = rendered.strip().lower()
    rendered = rendered.replace(".", "p")
    rendered = re.sub(r"[^a-z0-9_-]+", "-", rendered)
    return rendered.strip("-") or "none"


def build_log_name(args: argparse.Namespace) -> str:
    """Build an experiment filename from major run settings."""
    if args.log_name:
        name = Path(args.log_name).name
        return name if name.lower().endswith(".csv") else f"{name}.csv"

    grid = (
        f"b{int(args.block_size)}"
        if args.block_size is not None
        else f"auto-{_format_slug_value(args.detail_mode)}"
    )
    parts = [
        "pixelate-v4",
        f"mode-{_format_slug_value(args.mode)}",
        f"grid-{grid}",
        f"k-{int(args.palette_size)}",
        f"weight-{_format_slug_value(args.weight_mode)}",
        f"grad-{_format_slug_value(args.gradient_tolerance)}",
        "outline-on" if args.exclude_outline else "outline-off",
        (
            "usage-off"
            if args.disable_palette_usage_enforcement
            else "usage-on"
        ),
        "full" if args.preserve_size else "scaled",
    ]
    return "_".join(parts) + ".csv"


def _parameter_values(
    args: argparse.Namespace,
    *,
    effective_mode: str,
    effective_block_size: int | str,
) -> dict[str, object]:
    """Return all CLI settings in the stable logging schema."""
    return {
        "mode": effective_mode,
        "detail_mode": args.detail_mode,
        "weight_mode": args.weight_mode,
        "dither_method": args.dither_method,
        "palette_size": args.palette_size,
        "requested_block_size": (
            args.block_size if args.block_size is not None else "auto"
        ),
        "grid_source": (
            "fixed" if args.block_size is not None else "weighted"
        ),
        "preserve_size": args.preserve_size,
        "exclude_outline": args.exclude_outline,
        "palette_usage_enforcement": (
            not args.disable_palette_usage_enforcement
        ),
        "grid_candidates": args.grid_candidates,
        "edge_threshold": args.edge_threshold,
        "min_region_size": args.min_region_size,
        "segmentation_blur_ksize": args.segmentation_blur_ksize,
        "edge_dilate_iter": args.edge_dilate_iter,
        "area_cap": args.area_cap,
        "overcluster_factor": args.overcluster_factor,
        "gradient_tolerance": args.gradient_tolerance,
        "solid_max_distance": args.solid_max_distance,
        "minimum_dither_span": args.minimum_dither_span,
        "minimum_linearity": args.minimum_linearity,
        "maximum_reconstruction_rmse": args.maximum_reconstruction_rmse,
        "maximum_point_error": args.maximum_point_error,
        "minimum_endpoint_support_fraction": (
            args.minimum_endpoint_support_fraction
        ),
        "endpoint_quantile": args.endpoint_quantile,
        "forced_solid_penalty": args.forced_solid_penalty,
        "dither_penalty": args.dither_penalty,
        "matrix_size": args.matrix_size,
        "extreme_ratio_threshold": args.extreme_ratio_threshold,
        "outline_color_bin_size": args.outline_color_bin_size,
        "outline_minimum_edge_pixels": args.outline_minimum_edge_pixels,
        "outline_minimum_support_fraction": (
            args.outline_minimum_support_fraction
        ),
        "outline_color_tolerance": args.outline_color_tolerance,
        "outline_edge_coverage_threshold": (
            args.outline_edge_coverage_threshold
        ),
        "outline_fraction_threshold": args.outline_fraction_threshold,
        "outline_neighbor_radius": args.outline_neighbor_radius,
        "outline_spatial_penalty": args.outline_spatial_penalty,
        "random_state": args.random_state,
        "recursive": args.recursive,
        "overwrite": args.overwrite,
        "verbose": args.verbose,
        "suffix": args.suffix,
        "output_format": args.output_format,
        "no_log": args.no_log,
        "log_name_override": args.log_name or "",
        "block_size": effective_block_size,
    }


def _base_log_row(
    *,
    image_name: str,
    output_name: str,
    status: str,
    error: str,
    args: argparse.Namespace,
    effective_mode: str,
    effective_block_size: int | str,
) -> dict[str, object]:
    """Create a complete row with every declared CSV column."""
    row: dict[str, object] = {
        field: ""
        for field in LOG_FIELDNAMES
    }
    row.update({
        "image_name": image_name,
        "output_name": output_name,
        "status": status,
        "error": error,
    })
    row.update(_parameter_values(
        args,
        effective_mode=effective_mode,
        effective_block_size=effective_block_size,
    ))
    return row


def find_images(
    input_dir: Path,
    recursive: bool,
) -> list[Path]:
    """Find supported image files."""
    paths = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray:
    """Read one image in OpenCV BGR order."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def build_output_path(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
    mode: str,
    args: argparse.Namespace,
) -> Path:
    """Preserve relative subfolders and add a mode suffix when required."""
    relative = input_path.relative_to(input_dir)
    relative_stem = relative.with_suffix("")

    if args.mode == "all":
        file_name = (
            f"{relative_stem.name}_{mode}"
            f"{args.suffix}.{args.output_format}"
        )
    else:
        file_name = (
            f"{relative_stem.name}"
            f"{args.suffix}.{args.output_format}"
        )

    return output_dir / relative_stem.parent / file_name


def choose_block_size(
    image: np.ndarray,
    args: argparse.Namespace,
) -> int:
    """Choose a fixed block size or the unified weighted estimate."""
    if args.block_size is not None:
        block_size = int(args.block_size)
    else:
        block_size = estimate_block_size(
            image,
            mode=args.detail_mode,
            num_candidates=args.grid_candidates,
        )

    if block_size < 1:
        raise ValueError("block_size must be at least 1.")

    if block_size > min(image.shape[:2]):
        raise ValueError("block_size is larger than the image short side.")

    return block_size


def render_raw(
    image: np.ndarray,
    block_size: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Render the raw-pixel K-means baseline."""
    color_map, colors = derive_palette(
        image,
        palette_size=args.palette_size,
        random_state=args.random_state,
    )

    low_resolution = pixelate_image(
        image,
        block_size,
        color_map,
        colors,
    )

    output = expand_pixel_grid(
        low_resolution,
        block_size,
    )
    output = finalize_grid_output(
        output,
        block_size,
        preserve_size=args.preserve_size,
    )

    return output, {
        "requested_k": args.palette_size,
        "initial_k": len(colors),
        "compressed_k": len(colors),
        "solid_groups": len(colors),
        "dither_groups": 0,
    }


def render_region(
    image: np.ndarray,
    block_size: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Render the uncompressed region-balanced baseline."""
    color_map, colors, region_labels = derive_palette_region_balanced(
        image,
        palette_size=args.palette_size,
        edge_threshold=args.edge_threshold,
        weight_mode=args.weight_mode,
        area_cap=args.area_cap,
        min_region_size=args.min_region_size,
        segmentation_blur_ksize=args.segmentation_blur_ksize,
        edge_dilate_iter=args.edge_dilate_iter,
        random_state=args.random_state,
    )

    low_resolution = pixelate_image(
        image,
        block_size,
        color_map,
        colors,
    )

    output = expand_pixel_grid(
        low_resolution,
        block_size,
    )
    output = finalize_grid_output(
        output,
        block_size,
        preserve_size=args.preserve_size,
    )

    return output, {
        "requested_k": args.palette_size,
        "initial_k": len(colors),
        "compressed_k": len(colors),
        "solid_groups": len(colors),
        "dither_groups": 0,
        "segments": int(np.max(region_labels)) if region_labels.size else 0,
    }


def render_compressed(
    image: np.ndarray,
    block_size: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Render the palette-only compressed-family pipeline."""
    palette_result = derive_palette_compressed(
        image,
        palette_size=args.palette_size,
        edge_threshold=args.edge_threshold,
        weight_mode=args.weight_mode,
        area_cap=args.area_cap,
        min_region_size=args.min_region_size,
        segmentation_blur_ksize=args.segmentation_blur_ksize,
        edge_dilate_iter=args.edge_dilate_iter,
        exclude_outline=args.exclude_outline,
        outline_color_bin_size=args.outline_color_bin_size,
        outline_minimum_edge_pixels=(
            args.outline_minimum_edge_pixels
        ),
        outline_minimum_support_fraction=(
            args.outline_minimum_support_fraction
        ),
        overcluster_factor=args.overcluster_factor,
        gradient_tolerance=args.gradient_tolerance,
        solid_max_distance=args.solid_max_distance,
        minimum_dither_span=args.minimum_dither_span,
        minimum_linearity=args.minimum_linearity,
        maximum_reconstruction_rmse=args.maximum_reconstruction_rmse,
        maximum_point_error=args.maximum_point_error,
        minimum_endpoint_support_fraction=(
            args.minimum_endpoint_support_fraction
        ),
        endpoint_quantile=args.endpoint_quantile,
        forced_solid_penalty=args.forced_solid_penalty,
        dither_penalty=args.dither_penalty,
        random_state=args.random_state,
    )

    output = pixelate_image_with_compressed_palette(
        image,
        block_size,
        palette_result,
        dither_method=args.dither_method,
        matrix_size=args.matrix_size,
        extreme_ratio_threshold=args.extreme_ratio_threshold,
        apply_outline_exclusion=args.exclude_outline,
        outline_color_tolerance=args.outline_color_tolerance,
        outline_edge_coverage_threshold=(
            args.outline_edge_coverage_threshold
        ),
        outline_fraction_threshold=(
            args.outline_fraction_threshold
        ),
        outline_neighbor_radius=args.outline_neighbor_radius,
        outline_spatial_penalty=args.outline_spatial_penalty,
        ensure_palette_usage=(
            not args.disable_palette_usage_enforcement
        ),
    )

    output = finalize_grid_output(
        output,
        block_size,
        preserve_size=args.preserve_size,
    )

    return output, {
        "requested_k": palette_result.requested_k,
        "initial_k": palette_result.initial_k,
        "compressed_k": palette_result.compressed_k,
        "solid_groups": palette_result.solid_group_count,
        "dither_groups": palette_result.dither_group_count,
        "unique_palette_k": (
            palette_result.compression.unique_palette_size
        ),
        "forced_solid_groups": (
            palette_result.compression.forced_solid_count
        ),
        "segments": int(np.max(palette_result.region_labels))
        if palette_result.region_labels.size
        else 0,
    }


def render_mode(
    image: np.ndarray,
    block_size: int,
    mode: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Dispatch one render mode."""
    if mode == "raw":
        return render_raw(image, block_size, args)
    if mode == "region":
        return render_region(image, block_size, args)
    if mode == "compressed":
        return render_compressed(image, block_size, args)
    raise ValueError(f"Unknown mode: {mode}")


def process_one_image(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    """Process one source image and return schema-stable CSV rows."""
    image = read_image(input_path)
    height, width = image.shape[:2]
    block_size = choose_block_size(image, args)

    modes = (
        ["raw", "region", "compressed"]
        if args.mode == "all"
        else [args.mode]
    )

    rows: list[dict] = []

    for mode in modes:
        started = time.perf_counter()
        output_path = build_output_path(
            input_path,
            input_dir,
            output_dir,
            mode,
            args,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.overwrite:
            row = _base_log_row(
                image_name=input_path.name,
                output_name=output_path.name,
                status="skipped_exists",
                error="Use --overwrite to replace this file.",
                args=args,
                effective_mode=mode,
                effective_block_size=block_size,
            )
            row.update({
                "input_width": width,
                "input_height": height,
                "requested_k": args.palette_size,
                "elapsed_sec": round(
                    time.perf_counter() - started,
                    4,
                ),
            })
            rows.append(row)
            continue

        rendered, info = render_mode(
            image,
            block_size,
            mode,
            args,
        )

        if not cv2.imwrite(str(output_path), rendered):
            raise IOError(f"Could not write output: {output_path}")

        row = _base_log_row(
            image_name=input_path.name,
            output_name=output_path.name,
            status="ok",
            error="",
            args=args,
            effective_mode=mode,
            effective_block_size=block_size,
        )
        row.update({
            "input_width": width,
            "input_height": height,
            "output_width": int(rendered.shape[1]),
            "output_height": int(rendered.shape[0]),
            "requested_k": info.get("requested_k", ""),
            "initial_k": info.get("initial_k", ""),
            "compressed_k": info.get("compressed_k", ""),
            "solid_groups": info.get("solid_groups", ""),
            "dither_groups": info.get("dither_groups", ""),
            "unique_palette_k": info.get("unique_palette_k", ""),
            "forced_solid_groups": info.get(
                "forced_solid_groups",
                "",
            ),
            "segments": info.get("segments", ""),
            "elapsed_sec": round(
                time.perf_counter() - started,
                4,
            ),
        })
        rows.append(row)

    if args.verbose:
        print(
            f"  {input_path.name}: "
            f"block={block_size}, modes={len(modes)}"
        )

    return rows


def write_log(
    rows: list[dict],
    output_dir: Path,
    file_name: str,
) -> Path | None:
    """Write rows using the stable schema and return the log path."""
    if not rows:
        return None

    log_path = output_dir / Path(file_name).name
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=LOG_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: row.get(field, "")
                for field in LOG_FIELDNAMES
            })

    print(f"Log saved: {log_path.name}")
    return log_path


def process_folder(args: argparse.Namespace) -> None:
    """Batch-process an input folder."""
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(input_dir, args.recursive)
    if not images:
        raise SystemExit(f"No supported images found in: {input_dir}")

    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Images       : {len(images)}")
    print(f"Mode         : {args.mode}")
    print(f"Palette size : {args.palette_size}")
    print(f"Grid method  : weighted")
    print(f"Detail mode  : {args.detail_mode}")
    print("")

    rows: list[dict] = []
    failures = 0

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.relative_to(input_dir)}")

        try:
            rows.extend(
                process_one_image(
                    image_path,
                    input_dir,
                    output_dir,
                    args,
                )
            )
        except Exception as error:
            failures += 1
            print(f"  FAILED: {error}")
            row = _base_log_row(
                image_name=image_path.name,
                output_name="",
                status="failed",
                error=str(error),
                args=args,
                effective_mode=args.mode,
                effective_block_size="",
            )
            row["requested_k"] = args.palette_size
            rows.append(row)

    ok_count = sum(row["status"] == "ok" for row in rows)
    skipped_count = sum(row["status"] == "skipped_exists" for row in rows)

    print("")
    print(
        f"Finished. ok={ok_count}, "
        f"skipped={skipped_count}, failed={failures}"
    )

    if not args.no_log:
        write_log(
            rows,
            output_dir,
            build_log_name(args),
        )

    if failures:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch-run the Pixelate Everything Ver. 4 pipeline."
    )

    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--mode",
        choices=["raw", "region", "compressed", "all"],
        default="compressed",
    )

    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument(
        "--preserve_size",
        action="store_true",
        help=(
            "Keep the full block-resolution rendered image. By default, "
            "every block_size x block_size grid cell is area-averaged into "
            "one output pixel, including blended dither endpoint colors."
        ),
    )
    parser.add_argument(
        "--grid_candidates",
        type=int,
        default=20,
        help="Number of weighted grid-size candidates to evaluate.",
    )
    parser.add_argument(
        "--detail_mode",
        choices=["preserve", "balanced", "simplify"],
        default="balanced",
        help=(
            "All modes use the same weighted estimator. preserve gives the "
            "strongest preference to detailed regions; simplify gives the "
            "weakest, while still protecting detail."
        ),
    )

    parser.add_argument("--palette_size", type=int, default=32)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--edge_threshold", type=int, default=80)
    parser.add_argument(
        "--min_region_size",
        type=int,
        default=24,
        help=(
            "Discard segmented regions smaller than this pixel area. Increase "
            "it to suppress tiny outline fragments; decrease it to preserve "
            "small color regions."
        ),
    )
    parser.add_argument(
        "--segmentation_blur_ksize",
        type=int,
        default=5,
        help=(
            "Median-blur kernel before Canny segmentation. Odd values only; "
            "larger values suppress thin/noisy boundaries."
        ),
    )
    parser.add_argument(
        "--edge_dilate_iter",
        type=int,
        default=1,
        help=(
            "Dilation iterations for Canny boundaries before watershed. "
            "Larger values strengthen boundaries but can erase narrow regions."
        ),
    )
    parser.add_argument("--area_cap", type=int, default=2048)
    parser.add_argument(
        "--weight_mode",
        choices=["uniform", "area", "capped_area"],
        default="uniform",
        help="uniform means one vote per segmented region.",
    )

    parser.add_argument(
        "--exclude_outline",
        action="store_true",
        help=(
            "Detect the dominant color on segmentation edges and replace "
            "downsampled grid cells that are both edge-heavy and dominated "
            "by that outline color."
        ),
    )
    parser.add_argument(
        "--outline_color_bin_size",
        type=int,
        default=16,
        help=(
            "BGR quantization bin size used when finding the most frequent "
            "edge-pixel color."
        ),
    )
    parser.add_argument(
        "--outline_minimum_edge_pixels",
        type=int,
        default=16,
        help="Minimum Canny edge population required for outline detection.",
    )
    parser.add_argument(
        "--outline_minimum_support_fraction",
        type=float,
        default=0.03,
        help=(
            "Minimum fraction of edge pixels belonging to the modal color "
            "bin before outline exclusion is enabled."
        ),
    )
    parser.add_argument(
        "--outline_color_tolerance",
        type=float,
        default=12.0,
        help=(
            "CIELAB distance used to decide whether a rendered color matches "
            "the detected outline color."
        ),
    )
    parser.add_argument(
        "--outline_edge_coverage_threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum source-edge coverage inside a grid cell before that "
            "cell may be replaced."
        ),
    )
    parser.add_argument(
        "--outline_fraction_threshold",
        type=float,
        default=0.60,
        help=(
            "Minimum fraction of rendered pixels in a grid cell that must "
            "match the outline color."
        ),
    )
    parser.add_argument(
        "--outline_neighbor_radius",
        type=int,
        default=2,
        help="Maximum grid-cell radius searched for fallback colors.",
    )
    parser.add_argument(
        "--outline_spatial_penalty",
        type=float,
        default=0.35,
        help=(
            "Penalty for selecting farther fallback cells relative to color "
            "similarity with the source block."
        ),
    )

    parser.add_argument("--overcluster_factor", type=float, default=2.0)
    parser.add_argument(
        "--gradient_tolerance",
        type=float,
        default=1.0,
        help=(
            "Tolerance for recognizing K-means representatives as one "
            "ditherable color ramp. 1.0 keeps the base thresholds; values "
            "above 1.0 make compression looser, values below 1.0 make it "
            "stricter. Recommended tuning range: 0.5 to 2.0."
        ),
    )
    parser.add_argument("--solid_max_distance", type=float, default=4.0)
    parser.add_argument("--minimum_dither_span", type=float, default=6.0)
    parser.add_argument("--minimum_linearity", type=float, default=0.90)
    parser.add_argument(
        "--maximum_reconstruction_rmse",
        type=float,
        default=3.0,
    )
    parser.add_argument("--maximum_point_error", type=float, default=6.0)
    parser.add_argument(
        "--minimum_endpoint_support_fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument("--endpoint_quantile", type=float, default=0.05)
    parser.add_argument("--forced_solid_penalty", type=float, default=4.0)
    parser.add_argument("--dither_penalty", type=float, default=0.0)

    parser.add_argument(
        "--dither_method",
        choices=["ordered", "floyd_steinberg"],
        default="ordered",
    )
    parser.add_argument("--matrix_size", type=int, default=8)
    parser.add_argument("--extreme_ratio_threshold", type=float, default=0.08)
    parser.add_argument(
        "--disable_palette_usage_enforcement",
        action="store_true",
        help=(
            "Do not inject otherwise-unused palette entries into output "
            "pixels. This can remove isolated black dots, but the rendered "
            "image may use fewer unique colors than the selected palette."
        ),
    )

    parser.add_argument("--suffix", default="_px")
    parser.add_argument(
        "--output_format",
        choices=["png", "bmp", "webp"],
        default="png",
    )
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument(
        "--log_name",
        default=None,
        help=(
            "Optional CSV filename override. When omitted, the filename "
            "encodes mode and major experiment settings."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_folder(args)


if __name__ == "__main__":
    main()
