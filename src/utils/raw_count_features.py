"""CPI-safe summary features derived from raw mouse sensor counts.

The features in this module deliberately stay in integer sensor-count space.
They never convert counts to metres because that conversion requires the CPI
value that the amortizer is supposed to infer.
"""

from __future__ import annotations

import math

import numpy as np


RAW_COUNT_BIN_HZ = 125

RAW_COUNT_FEATURE_NAMES = (
    "raw_count_moving_bins",
    "raw_count_mean_per_moving_bin",
    "raw_count_p90_per_moving_bin",
    "raw_count_max_per_bin",
    "raw_count_path_per_second",
)

# Broad physical ranges used only for linear input normalization.  They cover
# the experiment's maximum three-second movement interval and leave generous
# headroom above the currently observed human values.
RAW_COUNT_NORMALIZATION_RANGES = {
    "raw_count_moving_bins": {"min": 0.0, "max": 400.0},
    "raw_count_mean_per_moving_bin": {"min": 0.0, "max": 256.0},
    "raw_count_p90_per_moving_bin": {"min": 0.0, "max": 512.0},
    "raw_count_max_per_bin": {"min": 0.0, "max": 1024.0},
    "raw_count_path_per_second": {"min": 0.0, "max": 20000.0},
}


def aggregate_count_events(
    event_end_times_s,
    count_deltas,
    *,
    movement_start_s: float,
    movement_end_s: float,
    bin_hz: int = RAW_COUNT_BIN_HZ,
) -> np.ndarray:
    """Aggregate count deltas into exact fixed-rate movement-time bins.

    ``event_end_times_s`` identifies the end of the interval represented by
    each ``(dx, dy)`` row.  An event exactly on a bin boundary belongs to the
    bin ending at that boundary.  Reaction-time events and post-click events
    are excluded before aggregation.
    """

    if int(bin_hz) != bin_hz or bin_hz <= 0:
        raise ValueError("bin_hz must be a positive integer")
    start = float(movement_start_s)
    end = float(movement_end_s)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("movement_end_s must be greater than movement_start_s")

    times = np.asarray(event_end_times_s, dtype=np.float64).reshape(-1)
    counts = np.asarray(count_deltas, dtype=np.float64)
    if counts.size == 0:
        counts = np.empty((0, 2), dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != 2:
        raise ValueError("count_deltas must have shape (n_events, 2)")
    if len(times) != len(counts):
        raise ValueError("event times and count deltas must have equal length")

    bin_width = 1.0 / int(bin_hz)
    duration = end - start
    n_bins = max(1, int(math.ceil(duration / bin_width - 1e-12)))
    output = np.zeros((n_bins, 2), dtype=np.float64)
    if len(times) == 0:
        return output

    finite = np.isfinite(times) & np.all(np.isfinite(counts), axis=1)
    tolerance = max(1e-12, duration * 1e-12)
    selected = finite & (times >= start - tolerance) & (times <= end + tolerance)
    if not np.any(selected):
        return output

    relative = np.clip(times[selected] - start, 0.0, duration)
    # End-of-interval convention: (0, 8 ms] is bin zero, (8, 16 ms] bin one.
    indices = np.ceil(relative / bin_width - 1e-12).astype(int) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    np.add.at(output, indices, counts[selected])
    return output


def raw_count_summary_features(
    binned_count_deltas,
    *,
    movement_duration_s: float,
) -> dict[str, float]:
    """Calculate five CPI-safe count summaries from fixed-rate bins."""

    counts = np.asarray(binned_count_deltas, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != 2:
        raise ValueError("binned_count_deltas must have shape (n_bins, 2)")
    duration = float(movement_duration_s)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("movement_duration_s must be positive")

    magnitude = np.linalg.norm(counts, axis=1)
    moving = magnitude > 0
    moving_magnitude = magnitude[moving]
    if moving_magnitude.size:
        mean_magnitude = float(np.mean(moving_magnitude))
        p90_magnitude = float(np.quantile(moving_magnitude, 0.9))
        max_magnitude = float(np.max(moving_magnitude))
    else:
        mean_magnitude = 0.0
        p90_magnitude = 0.0
        max_magnitude = 0.0

    return {
        "raw_count_moving_bins": float(np.count_nonzero(moving)),
        "raw_count_mean_per_moving_bin": mean_magnitude,
        "raw_count_p90_per_moving_bin": p90_magnitude,
        "raw_count_max_per_bin": max_magnitude,
        "raw_count_path_per_second": float(np.sum(magnitude) / duration),
    }


def count_event_summary_features(
    event_end_times_s,
    count_deltas,
    *,
    movement_start_s: float,
    movement_end_s: float,
    bin_hz: int = RAW_COUNT_BIN_HZ,
) -> dict[str, float]:
    """Aggregate raw events and calculate the standard count summaries."""

    binned = aggregate_count_events(
        event_end_times_s,
        count_deltas,
        movement_start_s=movement_start_s,
        movement_end_s=movement_end_s,
        bin_hz=bin_hz,
    )
    return raw_count_summary_features(
        binned,
        movement_duration_s=float(movement_end_s) - float(movement_start_s),
    )
