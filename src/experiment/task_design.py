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
) -> list[TrialTemplate]:
    """Reproduce the fixed-target sampling rule used by ``PnCGame.reset``.

    The target width is its diameter, hence ID = log2(D / W + 1).
    """
    rng = np.random.default_rng(seed)
    half_window = np.asarray(virtual_size_m, dtype=np.float64) / 2.0
    templates: list[TrialTemplate] = []

    for template_id in range(count):
        cursor = rng.uniform(-half_window, half_window)
        while True:
            fitts_id = float(rng.uniform(*id_range))
            target = rng.uniform(-half_window, half_window)
            distance = float(np.linalg.norm(target - cursor))
            radius = distance / (2.0**fitts_id - 1.0) / 2.0
            fits = np.all(target >= -half_window + radius) and np.all(
                target <= half_window - radius
            )
            if fits and distance >= radius:
                break

        templates.append(
            TrialTemplate(
                template_id=template_id,
                cursor_x_m=float(cursor[0]),
                cursor_y_m=float(cursor[1]),
                target_x_m=float(target[0]),
                target_y_m=float(target[1]),
                target_radius_m=float(radius),
                target_distance_m=float(distance),
                fitts_id=float(fitts_id),
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

