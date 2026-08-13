"""Matched final shared-CPI pair: speed/path-only versus CORE3.

Both profiles extend the existing shared-CPI two-step runner.  They differ
only in the retained curvature summaries and can therefore reuse one raw
20 Hz dataset generated from one hidden-CPI RL policy.
"""

from __future__ import annotations

import random
import sys

import numpy as np
import torch

from . import shared_cpi_2step_run as runner


CORE3_RETAINED = (
    "cursor_turn_energy_rad2",
    "cursor_early_mean_abs_turn_rad",
    "cursor_late_mean_abs_turn_rad",
)

PROFILE_REMOVED_FEATURES = {
    "speedpath": tuple(runner.CURVATURE_FEATURES),
    "core3": tuple(
        feature
        for feature in runner.CURVATURE_FEATURES
        if feature not in CORE3_RETAINED
    ),
}


def register_profiles() -> None:
    runner.FEATURE_PROFILES.update(PROFILE_REMOVED_FEATURES)


def _pop_seed(arguments: list[str]) -> int:
    option = "--pair-seed"
    if option not in arguments:
        return 20260813
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    seed = int(arguments[index + 1])
    del arguments[index : index + 2]
    return seed


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    register_profiles()
    seed = _pop_seed(arguments)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"FINAL_PAIR_SEED={seed}", flush=True)
    return runner.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
