"""Observable, scale-robust features for two-dimensional cursor trajectories.

The feature extractor deliberately accepts cursor positions and an optional target
position only.  In particular, it never receives mouse CPI, mouse gain, physical
hand motion, or raw-count distances that would require knowing CPI in advance.
"""

from __future__ import annotations

import numpy as np


CURVATURE_FEATURE_NAMES = (
    "cursor_total_abs_turn_rad",
    "cursor_net_turn_rad",
    "cursor_mean_abs_turn_rad",
    "cursor_p90_abs_turn_rad",
    "cursor_turn_energy_rad2",
    "cursor_max_deviation_normalized",
    "cursor_path_excess_ratio",
    "cursor_early_mean_abs_turn_rad",
    "cursor_middle_mean_abs_turn_rad",
    "cursor_late_mean_abs_turn_rad",
)

_EPS = 1e-12


def _as_finite_points(cursor_positions) -> np.ndarray:
    points = np.asarray(cursor_positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("cursor_positions must have shape (n_samples, 2)")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return points

    segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_length > _EPS))
    return points[keep]


def _resample_by_arc_length(points: np.ndarray, n_points: int) -> np.ndarray:
    if n_points < 3:
        raise ValueError("n_points must be at least 3")
    if len(points) < 2:
        return points

    segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(segment_length)))
    if arc_length[-1] <= _EPS:
        return points[:1]

    sample_arc = np.linspace(0.0, arc_length[-1], n_points)
    return np.column_stack(
        [np.interp(sample_arc, arc_length, points[:, axis]) for axis in range(2)]
    )


def _smooth_polyline(points: np.ndarray) -> np.ndarray:
    """Suppress single-count zigzags without changing the path endpoints."""
    if len(points) < 5:
        return points.copy()

    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    padded = np.pad(points, ((2, 2), (0, 0)), mode="edge")
    smoothed = np.column_stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)]
    )
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def _empty_features() -> dict[str, float]:
    return {name: 0.0 for name in CURVATURE_FEATURE_NAMES}


def cursor_curvature_features(
    cursor_positions,
    target_position=None,
    *,
    resample_points: int = 64,
) -> dict[str, float]:
    """Return dimensionless curvature descriptors from observable cursor positions.

    Points are resampled at equal arc-length intervals before turning angles are
    measured.  This makes the result substantially less sensitive to polling rate.
    Position scale, translation, and rotation do not change the angle features.
    Lateral deviation and path excess are normalized by the start-to-target
    distance (or by the start-to-end chord when no target is supplied).
    """
    points = _as_finite_points(cursor_positions)
    if len(points) < 2:
        return _empty_features()

    sampled = _smooth_polyline(
        _resample_by_arc_length(points, n_points=resample_points)
    )
    segments = np.diff(sampled, axis=0)
    segment_length = np.linalg.norm(segments, axis=1)
    valid_segments = segment_length > _EPS
    segments = segments[valid_segments]

    result = _empty_features()
    if len(segments) >= 2:
        first = segments[:-1]
        second = segments[1:]
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        dot = np.einsum("ij,ij->i", first, second)
        turn = np.arctan2(cross, dot)
        abs_turn = np.abs(turn)

        phases = np.array_split(abs_turn, 3)
        result.update(
            {
                "cursor_total_abs_turn_rad": float(np.sum(abs_turn)),
                "cursor_net_turn_rad": float(np.sum(turn)),
                "cursor_mean_abs_turn_rad": float(np.mean(abs_turn)),
                "cursor_p90_abs_turn_rad": float(np.percentile(abs_turn, 90)),
                "cursor_turn_energy_rad2": float(np.mean(np.square(turn))),
                "cursor_early_mean_abs_turn_rad": (
                    float(np.mean(phases[0])) if len(phases[0]) else 0.0
                ),
                "cursor_middle_mean_abs_turn_rad": (
                    float(np.mean(phases[1])) if len(phases[1]) else 0.0
                ),
                "cursor_late_mean_abs_turn_rad": (
                    float(np.mean(phases[2])) if len(phases[2]) else 0.0
                ),
            }
        )

    start = sampled[0]
    if target_position is None:
        reference_end = sampled[-1]
    else:
        reference_end = np.asarray(target_position, dtype=np.float64)
        if reference_end.shape != (2,) or not np.all(np.isfinite(reference_end)):
            raise ValueError("target_position must contain two finite coordinates")

    reference = reference_end - start
    reference_length = float(np.linalg.norm(reference))
    path_length = float(np.sum(np.linalg.norm(np.diff(sampled, axis=0), axis=1)))
    if reference_length > _EPS:
        relative = sampled - start
        signed_deviation = (
            reference[0] * relative[:, 1] - reference[1] * relative[:, 0]
        ) / (reference_length * reference_length)
        result["cursor_max_deviation_normalized"] = float(
            np.max(np.abs(signed_deviation))
        )
        result["cursor_path_excess_ratio"] = float(
            max(0.0, path_length / reference_length - 1.0)
        )

    return result
