"""Shared CPI-to-motion conversions for simulation and human experiments."""

from __future__ import annotations

import math


INCH_TO_METERS = 0.0254
DEFAULT_MOUSE_REFERENCE_CPI = 400.0


def positive_cpi(value, *, name: str = "cpi") -> float:
    cpi = float(value)
    if not math.isfinite(cpi) or cpi <= 0:
        raise ValueError(f"{name} must be a finite positive value, got {value!r}")
    return cpi


def physical_meters_per_count(cpi: float) -> float:
    """Physical desk distance represented by one raw mouse count."""
    return INCH_TO_METERS / positive_cpi(cpi)


def cursor_meters_per_count(
    mouse_gain: float,
    reference_cpi: float = DEFAULT_MOUSE_REFERENCE_CPI,
) -> float:
    """Virtual cursor distance per raw count at the calibrated reference CPI.

    Raw Input produces more counts for the same physical movement at higher
    CPI. Keeping this per-count scale fixed therefore makes a 1600-CPI mouse
    move the task cursor four times as far as a 400-CPI mouse for an identical
    hand displacement.
    """
    gain = float(mouse_gain)
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError(f"mouse_gain must be finite and positive, got {mouse_gain!r}")
    return gain * physical_meters_per_count(reference_cpi)


def effective_cursor_gain(
    cpi: float,
    mouse_gain: float,
    reference_cpi: float = DEFAULT_MOUSE_REFERENCE_CPI,
) -> float:
    """Return the cursor/hand displacement ratio at ``cpi``."""
    return (
        float(mouse_gain)
        * positive_cpi(cpi)
        / positive_cpi(reference_cpi, name="reference_cpi")
    )
