"""Pure task generation utilities; safe to test while RL training is running."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class TrialTemplate:
    template_id: int
    cursor_x_m: float
    cursor_y_m: float
    target_x_m: float
    target_y_m: float
    target_radius_m: float
    target_distance_m: float
    fitts_id: float

    def to_dict(self) -> dict:
        return asdict(self)


def make_trial_templates(
    count: int,
    virtual_size_m: tuple[float, float],
    id_range: tuple[float, float],
    seed: int,
    cursor_position_m: tuple[float, float] | None = None,
    target_radius_range_m: tuple[float, float] = (0.00125, 0.007),
    max_geometry_attempts: int = 100_000,
) -> list[TrialTemplate]:
    """Reproduce the fixed-target sampling rule used by ``PnCGame.reset``.

    The target width is its diameter, hence ID = log2(D / W + 1).
    """
    rng = np.random.default_rng(seed)
    half_window = np.asarray(virtual_size_m, dtype=np.float64) / 2.0
    templates: list[TrialTemplate] = []
    radius_min, radius_max = map(float, target_radius_range_m)
    if not (0.0 < radius_min <= radius_max):
        raise ValueError("target_radius_range_m must be positive and ordered")

    for template_id in range(count):
        # Sample ID once. Re-drawing ID after a rejected geometry biases the
        # realized distribution because difficult IDs are rejected more often.
        sampled_id = float(rng.uniform(*id_range))
        for _ in range(max_geometry_attempts):
            cursor = (
                rng.uniform(-half_window, half_window)
                if cursor_position_m is None
                else np.asarray(cursor_position_m, dtype=np.float64)
            )
            target = rng.uniform(-half_window, half_window)
            distance = float(np.linalg.norm(target - cursor))
            radius = distance / (2.0**sampled_id - 1.0) / 2.0
            fits = np.all(target >= -half_window + radius) and np.all(
                target <= half_window - radius
            )
            if (
                fits
                and distance >= radius
                and radius_min <= radius <= radius_max
            ):
                break
        else:
            raise RuntimeError(
                "Could not sample valid target geometry while preserving "
                f"Fitts ID={sampled_id:.6g} after {max_geometry_attempts} attempts"
            )

        templates.append(
            TrialTemplate(
                template_id=template_id,
                cursor_x_m=float(cursor[0]),
                cursor_y_m=float(cursor[1]),
                target_x_m=float(target[0]),
                target_y_m=float(target[1]),
                target_radius_m=float(radius),
                target_distance_m=float(distance),
                fitts_id=sampled_id,
            )
        )
    return templates


def counterbalanced_cpi_order(participant_id: str, cpi_values: list[int], seed: int) -> list[int]:
    """Choose one of the six 3-condition orders deterministically per participant."""
    orders = list(permutations(cpi_values))
    digest = hashlib.sha256(f"{seed}:{participant_id}".encode("utf-8")).digest()
    return list(orders[int.from_bytes(digest[:8], "little") % len(orders)])


def virtual_to_pixel(
    x_m: float,
    y_m: float,
    virtual_size_m: tuple[float, float],
    resolution_px: tuple[int, int],
) -> tuple[int, int]:
    width_m, height_m = virtual_size_m
    width_px, height_px = resolution_px
    x_px = round((x_m / width_m + 0.5) * width_px)
    y_px = round((0.5 - y_m / height_m) * height_px)
    return x_px, y_px


def fitts_id(distance_m: float, radius_m: float) -> float:
    return math.log2(distance_m / (2.0 * radius_m) + 1.0)


def angle_from_delta(delta: np.ndarray) -> float | None:
    """Return a vector heading in radians, or ``None`` for no movement."""
    vector = np.asarray(delta, dtype=np.float64)
    if float(np.linalg.norm(vector)) <= 0.0:
        return None
    return float(math.atan2(float(vector[1]), float(vector[0])))


def signed_angle_error(angle: float | None, reference: float | None) -> float | None:
    """Return the shortest signed angular difference in ``[-pi, pi)``."""
    if angle is None or reference is None:
        return None
    return float((angle - reference + math.pi) % (2.0 * math.pi) - math.pi)


def make_trial_stage_template(
    base_template: TrialTemplate,
    trial_stage: str,
    cursor_position_m: np.ndarray,
    virtual_size_m: tuple[float, float],
    start_target_radius_m: float = 0.006,
) -> TrialTemplate:
    """Build the start or task stage consistently for native and web clients."""
    cursor = np.asarray(cursor_position_m, dtype=np.float64)
    if trial_stage == "start":
        # The start target places the virtual cursor at the exact cursor
        # sampled together with this task, so task distance, radius and ID are
        # identical to the simulator rather than being recomputed or clamped.
        target = np.array(
            [base_template.cursor_x_m, base_template.cursor_y_m],
            dtype=np.float64,
        )
        radius = float(start_target_radius_m)
    elif trial_stage == "task":
        target = np.array(
            [base_template.target_x_m, base_template.target_y_m],
            dtype=np.float64,
        )
        radius = float(base_template.target_radius_m)
    else:
        raise ValueError(f"Unknown trial stage: {trial_stage}")

    distance = float(np.linalg.norm(target - cursor))
    return TrialTemplate(
        template_id=base_template.template_id,
        cursor_x_m=float(cursor[0]),
        cursor_y_m=float(cursor[1]),
        target_x_m=float(target[0]),
        target_y_m=float(target[1]),
        target_radius_m=radius,
        target_distance_m=distance,
        fitts_id=0.0 if distance <= 0 else fitts_id(distance, radius),
    )

