"""Shared-CPI summary-only runner retaining three CPI-consistent turn cues."""

from __future__ import annotations

from . import shared_cpi_2step_run as runner


RETAINED_CURVATURE_FEATURES = (
    "cursor_turn_energy_rad2",
    "cursor_early_mean_abs_turn_rad",
    "cursor_late_mean_abs_turn_rad",
)

runner.FEATURE_PROFILES["core3"] = tuple(
    feature
    for feature in runner.CURVATURE_FEATURES
    if feature not in RETAINED_CURVATURE_FEATURES
)


if __name__ == "__main__":
    raise SystemExit(runner.main())
