"""
Exact-budget recursive compression of palette representatives.
===============================================================

This module works only in palette space. It does not classify image regions
as gradients.

The central operation is:

    compress_representatives(colors, target_size=K, ...)

The input colors are recursively organized into a binary color-family tree.
For every tree node, the optimizer considers three representations:

1. split / keep detail
   Use the best representations selected for the two child subsets.

2. solid
   Replace the complete subset with one source-color medoid.

3. dither
   If the subset forms a sufficiently linear color ramp, replace it with two
   source-color endpoints and one interpolation ratio per member color.

Dynamic programming selects the minimum-error combination whose total palette
cost is exactly target_size. Therefore compression is selective rather than
being applied to every family that merely passes a local threshold.

OpenCV BGR order is used by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import cv2
import numpy as np


ColorOrder = Literal["BGR", "RGB", "LAB"]
GroupMode = Literal["solid", "dither"]


@dataclass(frozen=True)
class CompressedGroup:
    """One selected leaf group in the final exact-budget solution."""

    mode: GroupMode
    member_indices: np.ndarray

    # Used by a solid group.
    representative: np.ndarray | None = None

    # Used by a dither group.
    endpoint_a: np.ndarray | None = None
    endpoint_b: np.ndarray | None = None
    ratios: np.ndarray | None = None

    # Diagnostics in standard CIELAB units.
    linearity: float = 1.0
    endpoint_span: float = 0.0
    reconstruction_rmse: float = 0.0
    maximum_error: float = 0.0
    weighted_squared_error: float = 0.0
    forced_solid: bool = False

    @property
    def palette_cost(self) -> int:
        """Number of output palette entries used by this group."""
        return 1 if self.mode == "solid" else 2


@dataclass(frozen=True)
class CompressionResult:
    """Complete exact-budget compression result."""

    groups: tuple[CompressedGroup, ...]
    group_index_by_color: np.ndarray
    ratio_by_color: np.ndarray

    requested_size: int
    actual_size: int
    total_weighted_squared_error: float
    forced_solid_count: int

    def output_palette(self) -> np.ndarray:
        """Return the exact selected palette in group order."""
        output: list[np.ndarray] = []

        for group in self.groups:
            if group.mode == "solid":
                output.append(np.asarray(group.representative).reshape(3))
            else:
                output.append(np.asarray(group.endpoint_a).reshape(3))
                output.append(np.asarray(group.endpoint_b).reshape(3))

        if not output:
            return np.empty((0, 3), dtype=np.uint8)

        return np.vstack(output)

    @property
    def unique_palette_size(self) -> int:
        """Number of distinct output palette rows."""
        palette = self.output_palette()
        if len(palette) == 0:
            return 0
        return int(len(np.unique(palette, axis=0)))


@dataclass
class _TreeNode:
    """Internal binary family-tree node."""

    indices: np.ndarray
    left: _TreeNode | None = None
    right: _TreeNode | None = None
    solid_fit: dict | None = None
    dither_fit: dict | None = None

    @property
    def size(self) -> int:
        return int(len(self.indices))

    @property
    def is_leaf(self) -> bool:
        return self.left is None or self.right is None


@dataclass(frozen=True)
class _GroupSpec:
    """Internal unexpanded group description using unique-color indices."""

    mode: GroupMode
    indices: np.ndarray
    fit: dict


@dataclass(frozen=True)
class _Plan:
    """One dynamic-programming solution for a fixed palette cost."""

    score: float
    true_error: float
    groups: tuple[_GroupSpec, ...]
    forced_solid_count: int
    dither_group_count: int


# ---------------------------------------------------------------------------
# Validation and color conversion
# ---------------------------------------------------------------------------


def _validate_colors(colors: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(colors)

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("colors must have shape (N, 3).")
    if len(array) == 0:
        raise ValueError("colors must contain at least one color.")
    if not np.all(np.isfinite(array.astype(np.float64))):
        raise ValueError("colors contains NaN or infinite values.")

    return array.copy()


def _validate_weights(
    weights: Sequence[float] | None,
    count: int,
) -> np.ndarray:
    if weights is None:
        return np.ones(count, dtype=np.float64)

    array = np.asarray(weights, dtype=np.float64).reshape(-1)

    if len(array) != count:
        raise ValueError("weights must contain one value per color.")
    if not np.all(np.isfinite(array)):
        raise ValueError("weights contains NaN or infinite values.")
    if np.any(array < 0):
        raise ValueError("weights must be non-negative.")
    if float(np.sum(array)) <= 0:
        raise ValueError("weights must have positive total weight.")

    return array


def _to_lab(colors: np.ndarray, color_order: ColorOrder) -> np.ndarray:
    """Convert input colors to standard floating-point CIELAB."""
    if color_order == "LAB":
        return colors.astype(np.float64)

    normalized = np.clip(
        colors.astype(np.float32),
        0.0,
        255.0,
    ).reshape(1, -1, 3)
    normalized /= 255.0

    if color_order == "BGR":
        lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    elif color_order == "RGB":
        lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
    else:
        raise ValueError('color_order must be "BGR", "RGB", or "LAB".')

    return lab.reshape(-1, 3).astype(np.float64)


def _deduplicate_colors(
    source_colors: np.ndarray,
    source_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """
    Merge exactly duplicate input representatives.

    Returns:
        unique colors
        summed unique weights
        original -> unique mapping
        original-index arrays for every unique color
    """
    unique_colors, inverse = np.unique(
        source_colors,
        axis=0,
        return_inverse=True,
    )

    unique_weights = np.bincount(
        inverse,
        weights=source_weights,
        minlength=len(unique_colors),
    ).astype(np.float64)

    originals_by_unique = tuple(
        np.flatnonzero(inverse == unique_index).astype(np.int32)
        for unique_index in range(len(unique_colors))
    )

    return (
        unique_colors,
        unique_weights,
        inverse.astype(np.int32),
        originals_by_unique,
    )


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values * weights[:, None], axis=0) / float(np.sum(weights))


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    difference = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(difference * difference, axis=2))


def _maximum_pairwise_distance(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.max(_pairwise_distances(values)))


def _weighted_pca(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return weighted mean, principal axis, and PCA linearity."""
    mean = _weighted_mean(values, weights)
    centered = values - mean

    covariance = centered.T @ (centered * weights[:, None])
    covariance /= float(np.sum(weights))

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)

    axis = eigenvectors[:, order[0]]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / norm

    total_variance = float(np.sum(eigenvalues))
    linearity = (
        1.0
        if total_variance <= 1e-12
        else float(eigenvalues[0] / total_variance)
    )

    return mean, axis, linearity


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    quantile = float(np.clip(quantile, 0.0, 1.0))
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]

    cumulative = np.cumsum(sorted_weights)
    target = quantile * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


# ---------------------------------------------------------------------------
# Candidate fitting
# ---------------------------------------------------------------------------


def _fit_solid(
    indices: np.ndarray,
    lab_colors: np.ndarray,
    weights: np.ndarray,
    solid_max_distance: float,
) -> dict:
    """
    Fit one source-color medoid.

    A medoid is used instead of a free mean so selected output colors remain
    distinct source representatives whenever the input representatives are
    distinct.
    """
    subset = lab_colors[indices]
    subset_weights = weights[indices]

    differences = subset[:, None, :] - subset[None, :, :]
    squared = np.sum(differences * differences, axis=2)

    # Candidate j cost = sum_i weight_i * distance(i, j)^2.
    candidate_errors = np.sum(
        squared * subset_weights[:, None],
        axis=0,
    )
    local_medoid = int(np.argmin(candidate_errors))
    medoid_index = int(indices[local_medoid])

    distances = np.sqrt(squared[:, local_medoid])
    weighted_squared_error = float(candidate_errors[local_medoid])
    rmse = float(np.sqrt(
        weighted_squared_error / float(np.sum(subset_weights))
    ))
    maximum_error = float(np.max(distances))
    maximum_distance = _maximum_pairwise_distance(subset)

    return {
        "representative_index": medoid_index,
        "weighted_squared_error": weighted_squared_error,
        "reconstruction_rmse": rmse,
        "maximum_error": maximum_error,
        "maximum_distance": maximum_distance,
        "preferred": maximum_distance <= float(solid_max_distance),
    }


def _fit_dither_segment(
    indices: np.ndarray,
    lab_colors: np.ndarray,
    weights: np.ndarray,
    *,
    endpoint_quantile: float,
    minimum_dither_span: float,
    minimum_linearity: float,
    maximum_reconstruction_rmse: float,
    maximum_point_error: float,
    minimum_endpoint_support_fraction: float,
) -> dict:
    """Fit a two-source-endpoint dither segment to one family."""
    subset = lab_colors[indices]
    subset_weights = weights[indices]

    mean, axis, linearity = _weighted_pca(subset, subset_weights)
    projections = (subset - mean) @ axis

    low_value = _weighted_quantile(
        projections,
        subset_weights,
        endpoint_quantile,
    )
    high_value = _weighted_quantile(
        projections,
        subset_weights,
        1.0 - endpoint_quantile,
    )

    low_candidates = np.flatnonzero(projections <= low_value)
    high_candidates = np.flatnonzero(projections >= high_value)

    if len(low_candidates) == 0:
        low_candidates = np.array([int(np.argmin(projections))])
    if len(high_candidates) == 0:
        high_candidates = np.array([int(np.argmax(projections))])

    low_target = _weighted_mean(
        subset[low_candidates],
        subset_weights[low_candidates],
    )
    high_target = _weighted_mean(
        subset[high_candidates],
        subset_weights[high_candidates],
    )

    low_local = int(low_candidates[
        np.argmin(np.linalg.norm(subset[low_candidates] - low_target, axis=1))
    ])
    high_local = int(high_candidates[
        np.argmin(np.linalg.norm(subset[high_candidates] - high_target, axis=1))
    ])

    if low_local == high_local:
        low_local = int(np.argmin(projections))
        high_local = int(np.argmax(projections))

    endpoint_a_index = int(indices[low_local])
    endpoint_b_index = int(indices[high_local])
    endpoint_a = lab_colors[endpoint_a_index]
    endpoint_b = lab_colors[endpoint_b_index]

    direction = endpoint_b - endpoint_a
    span_squared = float(direction @ direction)
    endpoint_span = float(np.sqrt(span_squared))

    if span_squared <= 1e-12:
        ratios = np.zeros(len(subset), dtype=np.float64)
        reconstructed = np.repeat(endpoint_a[None, :], len(subset), axis=0)
    else:
        ratios = ((subset - endpoint_a) @ direction) / span_squared
        ratios = np.clip(ratios, 0.0, 1.0)
        reconstructed = endpoint_a[None, :] + ratios[:, None] * direction

    errors = np.linalg.norm(subset - reconstructed, axis=1)
    weighted_squared_error = float(np.sum(
        subset_weights * errors * errors
    ))
    rmse = float(np.sqrt(
        weighted_squared_error / float(np.sum(subset_weights))
    ))
    maximum_error = float(np.max(errors))

    total_weight = float(np.sum(subset_weights))
    low_support = float(np.sum(subset_weights[ratios <= 0.25])) / total_weight
    high_support = float(np.sum(subset_weights[ratios >= 0.75])) / total_weight

    valid = (
        len(indices) >= 3
        and endpoint_a_index != endpoint_b_index
        and endpoint_span >= float(minimum_dither_span)
        and linearity >= float(minimum_linearity)
        and rmse <= float(maximum_reconstruction_rmse)
        and maximum_error <= float(maximum_point_error)
        and low_support >= float(minimum_endpoint_support_fraction)
        and high_support >= float(minimum_endpoint_support_fraction)
    )

    return {
        "endpoint_a_index": endpoint_a_index,
        "endpoint_b_index": endpoint_b_index,
        "ratios": ratios,
        "linearity": float(linearity),
        "endpoint_span": endpoint_span,
        "reconstruction_rmse": rmse,
        "maximum_error": maximum_error,
        "weighted_squared_error": weighted_squared_error,
        "low_support_fraction": low_support,
        "high_support_fraction": high_support,
        "valid": bool(valid),
    }


# ---------------------------------------------------------------------------
# Family tree construction
# ---------------------------------------------------------------------------


def _split_two_means(
    lab_colors: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    *,
    max_iterations: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic weighted two-means split with a PCA fallback."""
    local_colors = lab_colors[indices]
    local_weights = weights[indices]

    distances = _pairwise_distances(local_colors)
    first, second = np.unravel_index(
        int(np.argmax(distances)),
        distances.shape,
    )

    center_a = local_colors[first].copy()
    center_b = local_colors[second].copy()
    assignments = np.full(len(indices), -1, dtype=np.int32)

    for _ in range(max_iterations):
        distance_a = np.linalg.norm(local_colors - center_a, axis=1)
        distance_b = np.linalg.norm(local_colors - center_b, axis=1)
        new_assignments = (distance_b < distance_a).astype(np.int32)

        if np.all(new_assignments == 0) or np.all(new_assignments == 1):
            break
        if np.array_equal(new_assignments, assignments):
            assignments = new_assignments
            break

        assignments = new_assignments
        mask_a = assignments == 0
        mask_b = assignments == 1
        center_a = _weighted_mean(local_colors[mask_a], local_weights[mask_a])
        center_b = _weighted_mean(local_colors[mask_b], local_weights[mask_b])

    if np.any(assignments == 0) and np.any(assignments == 1):
        return indices[assignments == 0], indices[assignments == 1]

    mean, axis, _ = _weighted_pca(local_colors, local_weights)
    projection = (local_colors - mean) @ axis
    order = np.argsort(projection)
    middle = max(1, min(len(indices) // 2, len(indices) - 1))
    return indices[order[:middle]], indices[order[middle:]]


def _build_tree(
    indices: np.ndarray,
    lab_colors: np.ndarray,
    weights: np.ndarray,
    *,
    solid_max_distance: float,
    endpoint_quantile: float,
    minimum_dither_span: float,
    minimum_linearity: float,
    maximum_reconstruction_rmse: float,
    maximum_point_error: float,
    minimum_endpoint_support_fraction: float,
) -> _TreeNode:
    node = _TreeNode(indices=indices.copy())
    node.solid_fit = _fit_solid(
        indices,
        lab_colors,
        weights,
        solid_max_distance,
    )

    if len(indices) >= 3:
        node.dither_fit = _fit_dither_segment(
            indices,
            lab_colors,
            weights,
            endpoint_quantile=endpoint_quantile,
            minimum_dither_span=minimum_dither_span,
            minimum_linearity=minimum_linearity,
            maximum_reconstruction_rmse=maximum_reconstruction_rmse,
            maximum_point_error=maximum_point_error,
            minimum_endpoint_support_fraction=minimum_endpoint_support_fraction,
        )

    if len(indices) == 1:
        return node

    left_indices, right_indices = _split_two_means(
        lab_colors,
        weights,
        indices,
    )

    node.left = _build_tree(
        left_indices,
        lab_colors,
        weights,
        solid_max_distance=solid_max_distance,
        endpoint_quantile=endpoint_quantile,
        minimum_dither_span=minimum_dither_span,
        minimum_linearity=minimum_linearity,
        maximum_reconstruction_rmse=maximum_reconstruction_rmse,
        maximum_point_error=maximum_point_error,
        minimum_endpoint_support_fraction=minimum_endpoint_support_fraction,
    )
    node.right = _build_tree(
        right_indices,
        lab_colors,
        weights,
        solid_max_distance=solid_max_distance,
        endpoint_quantile=endpoint_quantile,
        minimum_dither_span=minimum_dither_span,
        minimum_linearity=minimum_linearity,
        maximum_reconstruction_rmse=maximum_reconstruction_rmse,
        maximum_point_error=maximum_point_error,
        minimum_endpoint_support_fraction=minimum_endpoint_support_fraction,
    )
    return node


# ---------------------------------------------------------------------------
# Exact-budget dynamic programming
# ---------------------------------------------------------------------------


def _plan_key(plan: _Plan) -> tuple[float, int, int, int]:
    """Deterministic tie-break order."""
    return (
        float(plan.score),
        int(plan.forced_solid_count),
        int(plan.dither_group_count),
        int(len(plan.groups)),
    )


def _keep_better(
    options: dict[int, _Plan],
    budget: int,
    candidate: _Plan,
) -> None:
    current = options.get(budget)
    if current is None or _plan_key(candidate) < _plan_key(current):
        options[budget] = candidate


def _solve_exact_budget(
    node: _TreeNode,
    *,
    maximum_budget: int,
    forced_solid_penalty: float,
    dither_penalty: float,
) -> dict[int, _Plan]:
    """Return the best plan for every feasible palette cost at this node."""
    options: dict[int, _Plan] = {}

    solid_fit = node.solid_fit
    assert solid_fit is not None

    solid_true_error = float(solid_fit["weighted_squared_error"])
    forced = not bool(solid_fit["preferred"])
    solid_score = solid_true_error * (
        float(forced_solid_penalty) if forced else 1.0
    )

    _keep_better(
        options,
        1,
        _Plan(
            score=solid_score,
            true_error=solid_true_error,
            groups=(
                _GroupSpec(
                    mode="solid",
                    indices=node.indices.copy(),
                    fit=solid_fit,
                ),
            ),
            forced_solid_count=int(forced),
            dither_group_count=0,
        ),
    )

    if (
        node.dither_fit is not None
        and bool(node.dither_fit["valid"])
        and maximum_budget >= 2
    ):
        dither_error = float(node.dither_fit["weighted_squared_error"])
        _keep_better(
            options,
            2,
            _Plan(
                score=dither_error + float(dither_penalty),
                true_error=dither_error,
                groups=(
                    _GroupSpec(
                        mode="dither",
                        indices=node.indices.copy(),
                        fit=node.dither_fit,
                    ),
                ),
                forced_solid_count=0,
                dither_group_count=1,
            ),
        )

    if node.is_leaf:
        return options

    assert node.left is not None
    assert node.right is not None

    left_options = _solve_exact_budget(
        node.left,
        maximum_budget=min(maximum_budget, node.left.size),
        forced_solid_penalty=forced_solid_penalty,
        dither_penalty=dither_penalty,
    )
    right_options = _solve_exact_budget(
        node.right,
        maximum_budget=min(maximum_budget, node.right.size),
        forced_solid_penalty=forced_solid_penalty,
        dither_penalty=dither_penalty,
    )

    for left_budget, left_plan in left_options.items():
        for right_budget, right_plan in right_options.items():
            total_budget = left_budget + right_budget
            if total_budget > maximum_budget:
                continue

            combined = _Plan(
                score=left_plan.score + right_plan.score,
                true_error=left_plan.true_error + right_plan.true_error,
                groups=left_plan.groups + right_plan.groups,
                forced_solid_count=(
                    left_plan.forced_solid_count
                    + right_plan.forced_solid_count
                ),
                dither_group_count=(
                    left_plan.dither_group_count
                    + right_plan.dither_group_count
                ),
            )
            _keep_better(options, total_budget, combined)

    return options


# ---------------------------------------------------------------------------
# Result expansion
# ---------------------------------------------------------------------------


def _expand_group(
    spec: _GroupSpec,
    *,
    unique_colors: np.ndarray,
    originals_by_unique: tuple[np.ndarray, ...],
) -> CompressedGroup:
    original_indices = np.concatenate([
        originals_by_unique[int(unique_index)]
        for unique_index in spec.indices
    ]).astype(np.int32)

    if spec.mode == "solid":
        representative_index = int(spec.fit["representative_index"])
        representative = unique_colors[representative_index].copy()

        return CompressedGroup(
            mode="solid",
            member_indices=original_indices,
            representative=representative,
            endpoint_span=float(spec.fit["maximum_distance"]),
            reconstruction_rmse=float(spec.fit["reconstruction_rmse"]),
            maximum_error=float(spec.fit["maximum_error"]),
            weighted_squared_error=float(
                spec.fit["weighted_squared_error"]
            ),
            forced_solid=not bool(spec.fit["preferred"]),
        )

    endpoint_a_index = int(spec.fit["endpoint_a_index"])
    endpoint_b_index = int(spec.fit["endpoint_b_index"])

    ratio_by_unique = {
        int(unique_index): float(ratio)
        for unique_index, ratio in zip(
            spec.indices,
            np.asarray(spec.fit["ratios"], dtype=np.float64),
        )
    }

    ratios = np.empty(len(original_indices), dtype=np.float64)
    position = 0
    for unique_index in spec.indices:
        originals = originals_by_unique[int(unique_index)]
        count = len(originals)
        ratios[position:position + count] = ratio_by_unique[int(unique_index)]
        position += count

    return CompressedGroup(
        mode="dither",
        member_indices=original_indices,
        endpoint_a=unique_colors[endpoint_a_index].copy(),
        endpoint_b=unique_colors[endpoint_b_index].copy(),
        ratios=ratios,
        linearity=float(spec.fit["linearity"]),
        endpoint_span=float(spec.fit["endpoint_span"]),
        reconstruction_rmse=float(spec.fit["reconstruction_rmse"]),
        maximum_error=float(spec.fit["maximum_error"]),
        weighted_squared_error=float(spec.fit["weighted_squared_error"]),
        forced_solid=False,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compress_representatives(
    colors: Sequence[Sequence[float]],
    weights: Sequence[float] | None = None,
    *,
    target_size: int | None = None,
    color_order: ColorOrder = "BGR",
    solid_max_distance: float = 4.0,
    minimum_dither_span: float = 6.0,
    minimum_linearity: float = 0.90,
    maximum_reconstruction_rmse: float = 3.0,
    maximum_point_error: float = 6.0,
    minimum_endpoint_support_fraction: float = 0.05,
    endpoint_quantile: float = 0.05,
    forced_solid_penalty: float = 4.0,
    dither_penalty: float = 0.0,
) -> CompressionResult:
    """
    Selectively compress representatives under an exact palette budget.

    Parameters
    ----------
    colors:
        N x 3 input representatives.

    weights:
        Optional importance weight per representative. Region counts or
        capped-area weights can be used directly.

    target_size:
        Required number of output palette entries. It must be between 1 and
        the number of distinct input representatives. If omitted, the number
        of distinct input representatives is used, which performs no required
        compression.

    forced_solid_penalty:
        A solid merge is always available so every target size is feasible.
        If a subset exceeds solid_max_distance, its optimization score is
        multiplied by this penalty. This makes broad solid merges a last resort
        without making the exact budget impossible.

    Returns
    -------
    CompressionResult
        The output palette has exactly target_size distinct entries.
    """
    source_colors = _validate_colors(colors)
    source_weights = _validate_weights(weights, len(source_colors))

    if not 0.0 <= endpoint_quantile < 0.5:
        raise ValueError("endpoint_quantile must be in [0, 0.5).")
    if forced_solid_penalty < 1.0:
        raise ValueError("forced_solid_penalty must be >= 1.0.")

    (
        unique_colors,
        unique_weights,
        original_to_unique,
        originals_by_unique,
    ) = _deduplicate_colors(source_colors, source_weights)

    distinct_count = len(unique_colors)
    if target_size is None:
        target_size = distinct_count

    target_size = int(target_size)
    if target_size < 1:
        raise ValueError("target_size must be at least 1.")
    if target_size > distinct_count:
        raise ValueError(
            "target_size cannot exceed the number of distinct input colors: "
            f"target_size={target_size}, distinct_colors={distinct_count}."
        )

    lab_colors = _to_lab(unique_colors, color_order)
    root = _build_tree(
        np.arange(distinct_count, dtype=np.int32),
        lab_colors,
        unique_weights,
        solid_max_distance=solid_max_distance,
        endpoint_quantile=endpoint_quantile,
        minimum_dither_span=minimum_dither_span,
        minimum_linearity=minimum_linearity,
        maximum_reconstruction_rmse=maximum_reconstruction_rmse,
        maximum_point_error=maximum_point_error,
        minimum_endpoint_support_fraction=minimum_endpoint_support_fraction,
    )

    plans = _solve_exact_budget(
        root,
        maximum_budget=target_size,
        forced_solid_penalty=forced_solid_penalty,
        dither_penalty=dither_penalty,
    )

    selected = plans.get(target_size)
    if selected is None:
        raise RuntimeError(
            f"No exact compression plan found for target_size={target_size}."
        )

    groups = tuple(
        _expand_group(
            spec,
            unique_colors=unique_colors,
            originals_by_unique=originals_by_unique,
        )
        for spec in selected.groups
    )

    group_index_by_color = np.full(
        len(source_colors),
        -1,
        dtype=np.int32,
    )
    ratio_by_color = np.full(
        len(source_colors),
        np.nan,
        dtype=np.float64,
    )

    for group_index, group in enumerate(groups):
        group_index_by_color[group.member_indices] = group_index
        if group.mode == "dither":
            assert group.ratios is not None
            ratio_by_color[group.member_indices] = group.ratios

    if np.any(group_index_by_color < 0):
        raise RuntimeError("Internal error: some input colors were not assigned.")

    result = CompressionResult(
        groups=groups,
        group_index_by_color=group_index_by_color,
        ratio_by_color=ratio_by_color,
        requested_size=target_size,
        actual_size=sum(group.palette_cost for group in groups),
        total_weighted_squared_error=float(selected.true_error),
        forced_solid_count=int(selected.forced_solid_count),
    )

    if result.actual_size != target_size:
        raise RuntimeError(
            "Internal error: exact palette budget was not respected."
        )
    if result.unique_palette_size != target_size:
        raise RuntimeError(
            "Internal error: selected output palette contains duplicate colors."
        )

    return result


def describe_result(result: CompressionResult) -> str:
    """Return a compact readable summary."""
    lines = [
        f"requested_size={result.requested_size}",
        f"actual_size={result.actual_size}",
        f"unique_size={result.unique_palette_size}",
        f"groups={len(result.groups)}",
        f"forced_solid_groups={result.forced_solid_count}",
        (
            "weighted_squared_error="
            f"{result.total_weighted_squared_error:.3f}"
        ),
    ]

    for index, group in enumerate(result.groups):
        members = group.member_indices.tolist()
        if group.mode == "solid":
            lines.append(
                f"[{index}] solid members={members} "
                f"color={np.asarray(group.representative).tolist()} "
                f"forced={group.forced_solid}"
            )
        else:
            lines.append(
                f"[{index}] dither members={members} "
                f"A={np.asarray(group.endpoint_a).tolist()} "
                f"B={np.asarray(group.endpoint_b).tolist()} "
                f"ratios={np.round(group.ratios, 3).tolist()} " # type: ignore
                f"linearity={group.linearity:.3f} "
                f"rmse={group.reconstruction_rmse:.3f}"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    demo = np.array(
        [
            [15, 15, 15],
            [40, 40, 40],
            [75, 75, 75],
            [110, 110, 110],
            [150, 150, 150],
            [210, 210, 210],
            [30, 40, 210],
            [30, 210, 40],
        ],
        dtype=np.uint8,
    )

    compressed = compress_representatives(
        demo, # type: ignore
        target_size=5,
    )
    print(describe_result(compressed))
