"""Train a two-step amortizer with shared participant parameters and local CPI.

Each simulated participant contains several anonymous blocks.  The nine
non-CPI parameters are identical in every block, while CPI is sampled
independently for each block.  CPI values are training targets only; they are
never included in the behavioral input features or in the block labels.
"""

from __future__ import annotations

import argparse
import gc
import random
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from box import Box

from ..config.config import INF
from ..inference.pnc_dataset_2step import (
    PnCConditionalAmortizerTrainingDataset,
    PnCConditionalAmortizerValidationDataset,
)
from ..inference.pnc_infer_trainer_2step import PnCConditionalAmortizerTrainer
from ..utils.myutils import get_current_time_digit, save_dict_to_yaml


ROOT = Path(__file__).resolve().parents[2]

SHARED_PARAMETERS = (
    "theta_m",
    "theta_c",
    "rew_succ",
    "rew_fail",
    "time_penalty",
    "effort_cost",
    "mouse_rotation_gain_x",
    "mouse_rotation_gain_y",
    "mouse_plan_correction",
)
BLOCK_PARAMETERS = ("cpi",)
DATASET_CONTRACT = "movement_exact20_endpoint_clickonly_v3"

CURVATURE_FEATURES = (
    "cursor_total_abs_turn_rad",
    "cursor_net_turn_rad",
    "cursor_mean_abs_turn_rad",
    "cursor_p90_abs_turn_rad",
    "cursor_turn_energy_rad2",
    "cursor_max_deviation_normalized",
    "cursor_path_excess_ratio",
    "cursor_early_mean_abs_turn_rad",
    "cursor_middle_mean_abs_turn_rad",
    "cursor_late_mean_abs_turn_rad",
)

# Keep the original feature set available for exact baseline reproduction.
# CLEAN7 removes the two strongest human/simulation distribution mismatches and
# one member of the perfectly redundant total/mean-turn pair.
FEATURE_PROFILES = {
    "full": (),
    "clean7": (
        "cursor_total_abs_turn_rad",
        "cursor_p90_abs_turn_rad",
        "cursor_path_excess_ratio",
    ),
}
FEATURE_PROFILE_ADDITIONS = {}
FEATURE_PROFILE_RANGES = {}
FEATURE_PROFILE_METADATA = {}


def parse_checkpoint(value: str):
    if value in {"best", "last"}:
        return value
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "checkpoint must be 'best', 'last', or an integer training step"
        ) from error


def build_simulator_config(
    rl_model: str,
    rl_checkpoint,
    trajectory_downsample: int,
) -> Box:
    return Box({
        "model_name": rl_model,
        "ckpt": rl_checkpoint,
        "traj_downsample": trajectory_downsample,
        "simul_option": "default",
        "fix_per_param": [],
        "main_param": list(SHARED_PARAMETERS),
        "sub_param": list(BLOCK_PARAMETERS),
        "dataset_contract": DATASET_CONTRACT,
    })


def build_data_config(variant: str, feature_profile: str = "full") -> Box:
    config = deepcopy(INF.normalize.curvature)
    config.traj.ignore_traj = variant == "summary-only"
    removed = set(FEATURE_PROFILES[feature_profile])
    missing = removed - set(config.stat.list)
    if missing:
        raise ValueError(
            "Feature profile references unavailable statistics: "
            f"{', '.join(sorted(missing))}"
        )
    config.stat.list = [feature for feature in config.stat.list if feature not in removed]
    additions = list(FEATURE_PROFILE_ADDITIONS.get(feature_profile, ()))
    duplicates = set(additions) & set(config.stat.list)
    if duplicates:
        raise ValueError(
            "Feature profile adds statistics already present: "
            f"{', '.join(sorted(duplicates))}"
        )
    ranges = FEATURE_PROFILE_RANGES.get(feature_profile, {})
    missing_ranges = set(additions) - set(ranges)
    if missing_ranges:
        raise ValueError(
            "Added statistics are missing normalization ranges: "
            f"{', '.join(sorted(missing_ranges))}"
        )
    config.stat.list.extend(additions)
    for feature in additions:
        config.stat.range[feature] = deepcopy(ranges[feature])
    return config


def build_training_config(device: str | None = None) -> Box:
    # Keep the feature and trial encoders aligned with the independent CPI
    # amortizer.  Only the hierarchical output structure is changed.
    config = deepcopy(INF.training.default_cond)
    config.amortizer.encoder = deepcopy(INF.training.default.amortizer.encoder)
    config.amortizer.trial_encoder = deepcopy(
        INF.training.default.amortizer.trial_encoder
    )
    config.amortizer.loss_weight.main_param = [1.0] * len(SHARED_PARAMETERS)
    config.amortizer.loss_weight.sub_param = [1.0] * len(BLOCK_PARAMETERS)
    if device:
        config.amortizer.device = device
    return config


def close_stage_resources() -> None:
    plt.close("all")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-model", required=True)
    parser.add_argument("--rl-checkpoint", type=parse_checkpoint, default="best")
    parser.add_argument(
        "--variant",
        choices=("trajectory", "summary-only"),
        default="trajectory",
    )
    parser.add_argument(
        "--feature-profile",
        choices=tuple(FEATURE_PROFILES),
        default="full",
        help=(
            "Static-feature ablation profile. 'clean7' removes total absolute "
            "turn, P90 absolute turn, and path excess from the summary input."
        ),
    )
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-amortizer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-cpu", type=int, default=8)
    parser.add_argument("--train-main-params", type=int, default=43690)
    parser.add_argument("--blocks-per-participant", type=int, default=3)
    parser.add_argument("--trials-per-block", type=int, default=64)
    parser.add_argument("--data-repeat", type=int, default=1)
    parser.add_argument("--validation-users", type=int, default=100)
    parser.add_argument("--validation-trials", type=int, default=256)
    parser.add_argument("--trajectory-downsample", type=int, default=20)
    parser.add_argument("--amortizer-name", default="PTE_SHARED_CPI_2STEP_24FHD")
    parser.add_argument("--amortizer-iterations", type=int, default=200)
    parser.add_argument("--steps-per-iteration", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-frequency", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260819)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "data_cpu",
        "train_main_params",
        "blocks_per_participant",
        "trials_per_block",
        "data_repeat",
        "validation_users",
        "validation_trials",
        "trajectory_downsample",
        "amortizer_iterations",
        "steps_per_iteration",
        "batch_size",
        "save_frequency",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.blocks_per_participant != 3:
        raise SystemExit(
            "This experiment is defined for exactly three anonymous CPI blocks"
        )
    if args.feature_profile != "full" and args.variant != "summary-only":
        raise SystemExit(
            "Curvature feature ablations must use --variant summary-only; "
            "otherwise the trajectory encoder still receives the removed cues."
        )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    simulator_config = build_simulator_config(
        args.rl_model,
        args.rl_checkpoint,
        args.trajectory_downsample,
    )
    data_config = build_data_config(args.variant, args.feature_profile)
    training_config = build_training_config(args.device)

    removed_features = list(FEATURE_PROFILES[args.feature_profile])
    added_features = list(FEATURE_PROFILE_ADDITIONS.get(args.feature_profile, ()))
    retained_curvature = [
        feature for feature in CURVATURE_FEATURES if feature not in removed_features
    ]

    print("Shared CPI 2-step design", flush=True)
    print(f"  Random seed: {args.seed}", flush=True)
    print(f"  RL model: {args.rl_model} ({args.rl_checkpoint})", flush=True)
    print(f"  Variant: {args.variant}", flush=True)
    print(f"  Feature profile: {args.feature_profile}", flush=True)
    print(
        "  Removed summary features: "
        + (", ".join(removed_features) if removed_features else "none"),
        flush=True,
    )
    print(
        f"  Retained curvature features ({len(retained_curvature)}): "
        f"{', '.join(retained_curvature)}",
        flush=True,
    )
    print(
        "  Added summary features: "
        + (", ".join(added_features) if added_features else "none"),
        flush=True,
    )
    print(f"  Shared parameters ({len(SHARED_PARAMETERS)}): {', '.join(SHARED_PARAMETERS)}", flush=True)
    print(f"  Block-specific parameters: {', '.join(BLOCK_PARAMETERS)}", flush=True)
    print(f"  Anonymous blocks per participant: {args.blocks_per_participant}", flush=True)
    print("  True CPI is not an input feature or a grouping label.", flush=True)
    print(
        f"  Trajectory and motion-summary sampling: {args.trajectory_downsample} Hz",
        flush=True,
    )

    if args.dry_run:
        print("DRY_RUN_OK", flush=True)
        return 0

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this allocation")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if not args.skip_data:
        total_simulations = (
            args.train_main_params
            * args.blocks_per_participant
            * args.trials_per_block
        )
        print(
            "[1/2] Simulating shared-parameter dataset: "
            f"{total_simulations:,} trials",
            flush=True,
        )
        validation = PnCConditionalAmortizerValidationDataset(
            load_existing_data=False,
            simulator_config=simulator_config,
            normalize_config=data_config,
            total_user=args.validation_users,
            total_cond_per_user=args.blocks_per_participant,
            trial_per_user_cond=args.validation_trials,
        )
        validation._generate_raw_dataset(num_cpu=args.data_cpu)
        del validation
        close_stage_resources()

        training = PnCConditionalAmortizerTrainingDataset(
            load_existing_data=False,
            simulator_config=simulator_config,
            normalize_config=data_config,
            num_of_subparam_per_mainparam=args.blocks_per_participant,
            num_of_epsiode_per_param=args.trials_per_block,
        )
        training._generate_raw_dataset(
            n_main_param=args.train_main_params,
            repeat=args.data_repeat,
            num_cpu=args.data_cpu,
        )
        print(f"  Dataset: {Path(training.data_path).resolve()}", flush=True)
        del training
        close_stage_resources()
    else:
        print("[1/2] Reusing the existing shared-parameter dataset", flush=True)

    if not args.skip_amortizer:
        variant_label = "TRAJ" if args.variant == "trajectory" else "SUMMARY"
        model_name = (
            f"{args.amortizer_name}_{variant_label}_{get_current_time_digit()}"
        )
        print(f"[2/2] Training shared CPI amortizer: {model_name}", flush=True)
        print(
            f"SHARED_CPI_PROGRESS_TOTAL={args.amortizer_iterations}",
            flush=True,
        )
        trainer = PnCConditionalAmortizerTrainer(
            name=model_name,
            train_config=training_config,
            simul_config=simulator_config,
            data_config=data_config,
            train_subparams_per_main=args.blocks_per_participant,
            train_trials_per_subparam=args.trials_per_block,
            validation_users=args.validation_users,
            validation_subparams=args.blocks_per_participant,
            validation_trials=args.validation_trials,
        )
        design = {
            "schema_version": 2,
            "inference_structure": "shared_main_block_specific_cpi",
            "shared_parameters": list(SHARED_PARAMETERS),
            "block_specific_parameters": list(BLOCK_PARAMETERS),
            "blocks_per_participant": args.blocks_per_participant,
            "block_labels_are_inputs": False,
            "true_cpi_is_input": False,
            "variant": args.variant,
            "feature_profile": args.feature_profile,
            "removed_summary_features": removed_features,
            "added_summary_features": added_features,
            "retained_curvature_features": retained_curvature,
            "stat_features": list(data_config.stat.list),
            "trajectory_hz": args.trajectory_downsample,
            "motion_summary_hz": args.trajectory_downsample,
            "comparison_baseline": "independent PnCInferer fit",
            "random_seed": args.seed,
            "trajectory_preprocessing": (
                "crop reaction interval before exact fixed-rate resampling; "
                "preserve movement start and click endpoint"
            ),
            "truncated_trials": "retry up to 50 times, then fail dataset generation",
            "dataset_contract": DATASET_CONTRACT,
            "feature_profile_metadata": deepcopy(
                FEATURE_PROFILE_METADATA.get(args.feature_profile, {})
            ),
        }
        model_root = (
            ROOT / "data" / "amortizer_cond" / "models" / model_name
        )
        save_dict_to_yaml(design, str(model_root / "shared_fit_design.yaml"))
        trainer.train(
            n_iter=args.amortizer_iterations,
            step_per_iter=args.steps_per_iteration,
            batch_sz=args.batch_size,
            n_trial=args.trials_per_block,
            save_freq=args.save_frequency,
        )
        print(f"SHARED_CPI_COMPLETE={model_name}", flush=True)
        close_stage_resources()
    else:
        print("[2/2] Amortizer training skipped", flush=True)

    print(f"Artifacts: {(ROOT / 'data' / 'amortizer_cond').resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
