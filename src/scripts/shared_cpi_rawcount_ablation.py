"""Matched summary-only ablation: speed/path versus speed/path + raw counts."""

from __future__ import annotations

import sys

from . import shared_cpi_2step_run as runner
from ..utils.raw_count_features import (
    RAW_COUNT_BIN_HZ,
    RAW_COUNT_FEATURE_NAMES,
    RAW_COUNT_NORMALIZATION_RANGES,
)


DATASET_CONTRACT = "movement_exact20_endpoint_clickonly_rawcount125_v4"


def register_profiles() -> None:
    removed_curvature = tuple(runner.CURVATURE_FEATURES)
    runner.DATASET_CONTRACT = DATASET_CONTRACT
    runner.FEATURE_PROFILES.update({
        "speedpath": removed_curvature,
        "rawcount": removed_curvature,
    })
    runner.FEATURE_PROFILE_ADDITIONS["rawcount"] = RAW_COUNT_FEATURE_NAMES
    runner.FEATURE_PROFILE_RANGES["rawcount"] = RAW_COUNT_NORMALIZATION_RANGES
    runner.FEATURE_PROFILE_METADATA.update({
        "speedpath": {
            "raw_counts_used": False,
            "comparison_role": "matched reference",
        },
        "rawcount": {
            "raw_counts_used": True,
            "raw_count_bin_hz": RAW_COUNT_BIN_HZ,
            "raw_count_units": "integer sensor counts",
            "cpi_based_distance_conversion": False,
            "feature_names": list(RAW_COUNT_FEATURE_NAMES),
            "comparison_role": "raw-count ablation",
        },
    })


def main(argv=None) -> int:
    register_profiles()
    arguments = list(sys.argv[1:] if argv is None else argv)
    return runner.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
