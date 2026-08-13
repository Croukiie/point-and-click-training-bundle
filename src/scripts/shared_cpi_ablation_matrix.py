"""Controlled shared-CPI ablation matrix adapted from the colleague workflow.

The module extends the existing two-step runner without changing it.  Every
profile uses the same simulator, raw trials, 20 Hz preprocessing, parameter
partition, optimizer budget, and random seed.  Only the retained curvature
summary features differ.
"""

from __future__ import annotations

import json
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

ABLATION_ORDER = ("speedpath", "core3", "clean7", "full10")

PROFILE_REMOVED_FEATURES = {
    "speedpath": tuple(runner.CURVATURE_FEATURES),
    "core3": tuple(
        feature
        for feature in runner.CURVATURE_FEATURES
        if feature not in CORE3_RETAINED
    ),
    "clean7": tuple(runner.FEATURE_PROFILES["clean7"]),
    "full10": (),
}


def register_profiles() -> None:
    runner.FEATURE_PROFILES.update(PROFILE_REMOVED_FEATURES)


def retained_curvature(profile: str) -> tuple[str, ...]:
    removed = set(PROFILE_REMOVED_FEATURES[profile])
    return tuple(
        feature for feature in runner.CURVATURE_FEATURES if feature not in removed
    )


def matrix_manifest() -> dict:
    return {
        "schema_version": 1,
        "design": "matched_shared_cpi_curvature_ablation",
        "variant": "summary-only",
        "trajectory_hz": 20,
        "shared_parameters": list(runner.SHARED_PARAMETERS),
        "block_parameters": list(runner.BLOCK_PARAMETERS),
        "profiles": [
            {
                "index": index,
                "name": profile,
                "removed_curvature": list(PROFILE_REMOVED_FEATURES[profile]),
                "retained_curvature": list(retained_curvature(profile)),
                "retained_count": len(retained_curvature(profile)),
            }
            for index, profile in enumerate(ABLATION_ORDER)
        ],
    }


def _pop_option(arguments: list[str], option: str, default: str) -> str:
    if option not in arguments:
        return default
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    value = arguments[index + 1]
    del arguments[index : index + 2]
    return value


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    register_profiles()

    if arguments == ["--print-matrix"]:
        print(json.dumps(matrix_manifest(), indent=2))
        return 0

    seed = int(_pop_option(arguments, "--ablation-seed", "20260813"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"ABLATION_SEED={seed}", flush=True)
    return runner.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
