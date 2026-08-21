"""Matched CPI-visible RL ablation: speed/path-only versus CORE3."""

from __future__ import annotations

import sys

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


def _translate_pair_seed(arguments: list[str]) -> list[str]:
    """Keep compatibility with existing Slurm scripts without double seeding."""
    arguments = list(arguments)
    option = "--pair-seed"
    if option not in arguments:
        return arguments
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise SystemExit(f"{option} requires a value")
    seed = arguments[index + 1]
    del arguments[index : index + 2]
    if "--seed" in arguments:
        raise SystemExit("Use either --pair-seed or --seed, not both")
    arguments.extend(("--seed", seed))
    return arguments


def main(argv=None) -> int:
    register_profiles()
    arguments = list(sys.argv[1:] if argv is None else argv)
    return runner.main(_translate_pair_seed(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
