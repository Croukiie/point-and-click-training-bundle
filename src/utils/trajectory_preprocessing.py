"""Shared endpoint-preserving trajectory preprocessing for simulation and humans."""

from __future__ import annotations

import numpy as np


def endpoint_preserving_grid(duration_s: float, sampling_hz: float) -> np.ndarray:
    """Return an exact fixed-rate grid with an explicit terminal sample.

    Regular samples are separated by ``1 / sampling_hz``.  The final interval
    may be shorter so the click timestamp is represented exactly.
    """
    duration_s = float(duration_s)
    sampling_hz = float(sampling_hz)
    if sampling_hz <= 0:
        raise ValueError("sampling_hz must be positive")
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")
    if duration_s == 0:
        return np.array([0.0], dtype=np.float64)

    step = 1.0 / sampling_hz
    grid = np.arange(0.0, duration_s, step, dtype=np.float64)
    if grid.size == 0 or grid[0] != 0.0:
        grid = np.insert(grid, 0, 0.0)
    if np.isclose(grid[-1], duration_s, rtol=0.0, atol=1e-12):
        grid[-1] = duration_s
    else:
        grid = np.append(grid, duration_s)
    return grid


def resample_movement(
    timestamps_s,
    values,
    *,
    movement_start_s: float,
    movement_end_s: float,
    sampling_hz: float,
    start_value=None,
    end_value=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop to movement, resample, and preserve the start/click endpoints.

    ``timestamps_s`` use the source clock. Returned timestamps start at zero.
    ``values`` may contain one or more columns.
    """
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    if len(timestamps) != len(samples):
        raise ValueError("timestamps_s and values must have the same length")
    if samples.ndim != 2:
        raise ValueError("values must have shape (n_samples, n_features)")

    start = float(movement_start_s)
    end = float(movement_end_s)
    if end < start:
        raise ValueError("movement_end_s must not precede movement_start_s")

    finite = np.isfinite(timestamps) & np.all(np.isfinite(samples), axis=1)
    timestamps = timestamps[finite]
    samples = samples[finite]
    if timestamps.size == 0:
        raise ValueError("trajectory contains no finite sample")

    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    samples = samples[order]

    def boundary_value(moment: float, explicit):
        if explicit is not None:
            value = np.asarray(explicit, dtype=np.float64).reshape(-1)
            if value.size != samples.shape[1]:
                raise ValueError("explicit endpoint has the wrong feature count")
            return value
        return np.array(
            [np.interp(moment, timestamps, samples[:, i]) for i in range(samples.shape[1])],
            dtype=np.float64,
        )

    start_sample = boundary_value(start, start_value)
    end_sample = boundary_value(end, end_value)
    interior = (timestamps > start) & (timestamps < end)
    cropped_t = np.concatenate(([start], timestamps[interior], [end]))
    cropped_values = np.vstack((start_sample, samples[interior], end_sample))

    # Keep the last value for duplicate source timestamps.
    unique_t = []
    unique_values = []
    for timestamp, value in zip(cropped_t, cropped_values):
        if unique_t and timestamp <= unique_t[-1]:
            unique_t[-1] = float(timestamp)
            unique_values[-1] = value
        else:
            unique_t.append(float(timestamp))
            unique_values.append(value)
    source_t = np.asarray(unique_t, dtype=np.float64)
    source_values = np.asarray(unique_values, dtype=np.float64)

    relative_t = endpoint_preserving_grid(end - start, sampling_hz)
    absolute_t = relative_t + start
    resampled = np.column_stack(
        [np.interp(absolute_t, source_t, source_values[:, i]) for i in range(source_values.shape[1])]
    )
    resampled[0] = start_sample
    resampled[-1] = end_sample
    return relative_t, resampled


def resample_feature_frame(
    frame,
    feature_names,
    sampling_hz: float,
    *,
    timestamp_key: str = "timestamp",
    timediff_key: str = "timediff",
) -> np.ndarray:
    """Resample dataframe-like trajectory features with the shared contract."""
    source_t = np.asarray(frame[timestamp_key], dtype=np.float64)
    if source_t.size == 0:
        raise ValueError("trajectory is empty")
    continuous = [name for name in feature_names if name != timediff_key]
    source_values = np.column_stack(
        [np.asarray(frame[name], dtype=np.float64) for name in continuous]
    )
    new_t, new_values = resample_movement(
        source_t,
        source_values,
        movement_start_s=float(source_t[0]),
        movement_end_s=float(source_t[-1]),
        sampling_hz=sampling_hz,
        start_value=source_values[0],
        end_value=source_values[-1],
    )
    values_by_name = {
        name: new_values[:, index] for index, name in enumerate(continuous)
    }
    values_by_name[timediff_key] = np.insert(np.diff(new_t), 0, 0.0)
    return np.column_stack([values_by_name[name] for name in feature_names])
