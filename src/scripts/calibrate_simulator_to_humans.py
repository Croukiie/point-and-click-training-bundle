"""Rank simulated parameter profiles by similarity to human pilot behavior.

This is a calibration audit, not an inference step.  It reuses an existing
simulation dataset, scores absolute behavioral levels and within-profile CPI
trends, then proposes an exploratory restricted prior from the best profiles.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PARAMETER_NAMES = (
    "theta_m",
    "theta_c",
    "rew_succ",
    "rew_fail",
    "time_penalty",
    "effort_cost",
    "mouse_rotation_gain_x",
    "mouse_rotation_gain_y",
    "mouse_plan_correction",
    "cpi",
)

LEVEL_FEATURES = (
    "movement_time",
    "target_hit",
    "click_distance",
    "max_cursor_speed",
    "cursor_total_abs_turn_rad",
    "cursor_path_excess_ratio",
)

TREND_FEATURES = (
    "max_cursor_speed",
    "cursor_total_abs_turn_rad",
    "cursor_path_excess_ratio",
)

FEATURE_LABELS = {
    "movement_time": "Movement time (s)",
    "target_hit": "Target hit rate",
    "click_distance": "Click distance (m)",
    "max_cursor_speed": "Maximum cursor speed (m/s)",
    "cursor_total_abs_turn_rad": "Total absolute turn (rad)",
    "cursor_path_excess_ratio": "Cursor path excess ratio",
}

# Prevent two-person pilot standard deviations from making the score unstable.
SCALE_FLOORS = {
    "movement_time": 0.15,
    "target_hit": 0.05,
    "click_distance": 0.003,
    "max_cursor_speed": 0.20,
    "cursor_total_abs_turn_rad": 0.40,
    "cursor_path_excess_ratio": 0.08,
}

LEVEL_WEIGHTS = {
    "movement_time": 1.5,
    "target_hit": 2.0,
    "click_distance": 2.0,
    "max_cursor_speed": 1.0,
    "cursor_total_abs_turn_rad": 1.0,
    "cursor_path_excess_ratio": 1.0,
}

TREND_SCALES = {
    "max_cursor_speed": 0.25,
    "cursor_total_abs_turn_rad": 0.50,
    "cursor_path_excess_ratio": 0.10,
}


def progress(percent: int, message: str) -> None:
    percent = int(np.clip(percent, 0, 100))
    width = 30
    filled = round(width * percent / 100)
    bar = "#" * filled + "-" * (width - filled)
    print(f"CALIBRATION [{bar}] {percent:3d}%  {message}", flush=True)


def find_dataset(root: Path, *, smallest: bool) -> Path:
    candidates = list((root / "data" / "amortizer_cond" / "dataset").glob(
        "**/vd_*.npz"
    ))
    if not candidates:
        raise FileNotFoundError(
            "No shared validation dataset was found under "
            "data/amortizer_cond/dataset"
        )
    candidates.sort(key=lambda path: (path.stat().st_size, path.stat().st_mtime))
    return candidates[0] if smallest else candidates[-1]


def read_feature_names(dataset_path: Path) -> list[str]:
    metadata = dataset_path.with_suffix(".yaml")
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata}")
    with metadata.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    names = list(config["stat"])
    required = {*PARAMETER_NAMES, *LEVEL_FEATURES}
    missing = required - set(names)
    if missing:
        raise ValueError(f"Dataset is missing fields: {sorted(missing)}")
    return names


def load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(dataset_path, allow_pickle=True) as archive:
        params = np.asarray(archive["params"], dtype=np.float64)
        stat_data = np.asarray(archive["stat_data"], dtype=np.float64)
    if params.ndim != 2 or stat_data.ndim != 3:
        raise ValueError(
            f"Unexpected shapes: params={params.shape}, stat_data={stat_data.shape}"
        )
    if len(params) != len(stat_data):
        raise ValueError("params and stat_data contain different condition counts")
    if params.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(
            f"Expected {len(PARAMETER_NAMES)} parameters, found {params.shape[1]}"
        )
    return params, stat_data


def group_ids_from_shared_parameters(params: np.ndarray) -> np.ndarray:
    key_to_id: dict[tuple[float, ...], int] = {}
    result = np.empty(len(params), dtype=np.int64)
    for index, row in enumerate(params[:, :-1]):
        key = tuple(np.round(row, decimals=6))
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        result[index] = key_to_id[key]
    return result


def condition_table(
    params: np.ndarray,
    stats: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    del params  # Physical values are read from stat_data to avoid scale assumptions.
    rows: dict[str, Any] = {}
    for parameter in PARAMETER_NAMES:
        rows[parameter] = np.nanmean(
            stats[:, :, feature_names.index(parameter)], axis=1
        )
    for feature in LEVEL_FEATURES:
        rows[feature] = np.nanmean(
            stats[:, :, feature_names.index(feature)], axis=1
        )
    return pd.DataFrame(rows)


def validate_targets(targets: pd.DataFrame) -> pd.DataFrame:
    required = {"cpi", "feature", "human_mean", "human_sd"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"Human target table is missing columns: {sorted(missing)}")
    targets = targets.copy()
    targets["cpi"] = pd.to_numeric(targets["cpi"], errors="raise")
    targets["human_mean"] = pd.to_numeric(targets["human_mean"], errors="raise")
    targets["human_sd"] = pd.to_numeric(targets["human_sd"], errors="coerce").fillna(0)
    missing_features = set(LEVEL_FEATURES) - set(targets["feature"])
    if missing_features:
        raise ValueError(
            f"Human target table is missing features: {sorted(missing_features)}"
        )
    return targets.sort_values(["feature", "cpi"]).reset_index(drop=True)


def interpolated_target(
    targets: pd.DataFrame,
    feature: str,
    cpi_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    subset = targets[targets["feature"] == feature].sort_values("cpi")
    x = subset["cpi"].to_numpy(dtype=float)
    mean = np.interp(cpi_values, x, subset["human_mean"].to_numpy(dtype=float))
    sd = np.interp(cpi_values, x, subset["human_sd"].to_numpy(dtype=float))
    scale = np.maximum(sd, SCALE_FLOORS[feature])
    return mean, scale


def add_level_scores(conditions: pd.DataFrame, targets: pd.DataFrame) -> None:
    cpi_values = conditions["cpi"].to_numpy(dtype=float)
    weighted_parts = []
    weights = []
    for feature in LEVEL_FEATURES:
        expected, scale = interpolated_target(targets, feature, cpi_values)
        residual = (conditions[feature].to_numpy(dtype=float) - expected) / scale
        conditions[f"z_{feature}"] = residual
        weight = LEVEL_WEIGHTS[feature]
        weighted_parts.append(weight * residual ** 2)
        weights.append(weight)
    conditions["level_score"] = np.sum(weighted_parts, axis=0) / sum(weights)
    conditions["hard_level_pass"] = (
        conditions["movement_time"].between(0.85, 1.35)
        & (conditions["target_hit"] >= 0.90)
        & (conditions["click_distance"] <= 0.005)
    )


def endpoint_delta(group: pd.DataFrame, feature: str) -> float:
    ordered = group.sort_values("cpi")
    cpi = ordered["cpi"].to_numpy(dtype=float)
    values = ordered[feature].to_numpy(dtype=float)
    mask = np.isfinite(cpi) & np.isfinite(values)
    if mask.sum() < 2 or np.ptp(cpi[mask]) <= 0:
        return float("nan")
    slope = np.polyfit(cpi[mask], values[mask], deg=1)[0]
    return float(slope * 1200.0)


def human_endpoint_delta(targets: pd.DataFrame, feature: str) -> float:
    subset = targets[targets["feature"] == feature].sort_values("cpi")
    cpi = subset["cpi"].to_numpy(dtype=float)
    values = subset["human_mean"].to_numpy(dtype=float)
    slope = np.polyfit(cpi, values, deg=1)[0]
    return float(slope * 1200.0)


def candidate_table(
    conditions: pd.DataFrame,
    targets: pd.DataFrame,
    trend_weight: float,
) -> pd.DataFrame:
    human_deltas = {
        feature: human_endpoint_delta(targets, feature)
        for feature in TREND_FEATURES
    }
    candidates = []
    for group_id, group in conditions.groupby("group_id", sort=True):
        row: dict[str, Any] = {
            "group_id": int(group_id),
            "n_conditions": int(len(group)),
            "cpi_min": float(group["cpi"].min()),
            "cpi_max": float(group["cpi"].max()),
            "level_score": float(group["level_score"].mean()),
            "hard_level_pass": bool(group["hard_level_pass"].all()),
        }
        for parameter in PARAMETER_NAMES[:-1]:
            row[parameter] = float(group[parameter].iloc[0])

        trend_parts = []
        trend_signs = []
        for feature in TREND_FEATURES:
            simulated_delta = endpoint_delta(group, feature)
            row[f"delta_{feature}"] = simulated_delta
            target_delta = human_deltas[feature]
            row[f"human_delta_{feature}"] = target_delta
            if np.isfinite(simulated_delta):
                trend_parts.append(
                    ((simulated_delta - target_delta) / TREND_SCALES[feature]) ** 2
                )
                trend_signs.append(
                    np.sign(simulated_delta) == np.sign(target_delta)
                    or abs(target_delta) < 1e-12
                )

        trend_score = float(np.mean(trend_parts)) if trend_parts else np.nan
        row["trend_score"] = trend_score
        row["hard_trend_pass"] = bool(trend_signs and all(trend_signs))
        row["total_score"] = row["level_score"] + (
            trend_weight * trend_score if np.isfinite(trend_score) else 0.0
        )
        row["hard_pass"] = row["hard_level_pass"] and (
            row["hard_trend_pass"] if trend_parts else True
        )
        candidates.append(row)
    result = pd.DataFrame(candidates).sort_values(
        ["total_score", "level_score"], ignore_index=True
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def proposed_prior(
    candidates: pd.DataFrame,
    dataset_path: Path,
    keep_fraction: float,
    minimum_keep: int,
) -> dict[str, Any]:
    keep_count = min(
        len(candidates),
        max(minimum_keep, int(math.ceil(len(candidates) * keep_fraction))),
    )
    selected = candidates.head(keep_count)
    ranges = {}
    for parameter in PARAMETER_NAMES[:-1]:
        values = selected[parameter].to_numpy(dtype=float)
        ranges[parameter] = {
            "type": "uniform",
            "min": float(np.nanquantile(values, 0.05)),
            "max": float(np.nanquantile(values, 0.95)),
            "selected_mean": float(np.nanmean(values)),
        }
    ranges["cpi"] = {"type": "fixed_blocks", "values": [400, 800, 1600]}
    return {
        "schema_version": 1,
        "status": "exploratory_pilot_prior_do_not_treat_as_final",
        "source_dataset": str(dataset_path.resolve()),
        "candidate_groups": int(len(candidates)),
        "selected_groups": int(keep_count),
        "keep_fraction_requested": float(keep_fraction),
        "selection": "lowest human-calibration total_score",
        "parameter_ranges": ranges,
    }


def plot_score_distribution(candidates: pd.DataFrame, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    finite = candidates["total_score"].replace([np.inf, -np.inf], np.nan).dropna()
    axis.hist(finite, bins=min(50, max(5, round(np.sqrt(len(finite))))), color="#2878b5")
    if len(finite):
        axis.axvline(float(finite.min()), color="#c44e52", linewidth=2, label="best")
        axis.legend()
    axis.set_xlabel("Human-calibration score (lower is better)")
    axis.set_ylabel("Parameter groups")
    axis.set_title("Simulation profile calibration scores")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_trends(
    candidates: pd.DataFrame,
    conditions: pd.DataFrame,
    targets: pd.DataFrame,
    destination: Path,
    top_plot: int,
) -> None:
    selected_ids = set(candidates.head(top_plot)["group_id"].astype(int))
    selected = conditions[conditions["group_id"].isin(selected_ids)]
    best_id = int(candidates.iloc[0]["group_id"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for axis, feature in zip(axes.flat, LEVEL_FEATURES):
        target = targets[targets["feature"] == feature].sort_values("cpi")
        axis.errorbar(
            target["cpi"],
            target["human_mean"],
            yerr=np.maximum(target["human_sd"], 0),
            color="#111111",
            marker="o",
            linewidth=2.5,
            capsize=3,
            label="human target",
            zorder=5,
        )
        for group_id, group in selected.groupby("group_id", sort=False):
            group = group.sort_values("cpi")
            is_best = int(group_id) == best_id
            axis.plot(
                group["cpi"],
                group[feature],
                color="#c44e52" if is_best else "#2878b5",
                linewidth=2.5 if is_best else 1.0,
                alpha=1.0 if is_best else 0.18,
                label="best simulation" if is_best else None,
            )
        axis.set_title(FEATURE_LABELS[feature])
        axis.set_xlabel("CPI")
        axis.set_xticks([400, 800, 1600])
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Human targets and top calibrated simulation profiles", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path.cwd())
    parser.add_argument("--human-targets", type=Path, required=True)
    parser.add_argument("--simulation-dataset", type=Path)
    parser.add_argument("--prefer-smallest-dataset", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-keep", type=int, default=20)
    parser.add_argument("--trend-weight", type=float, default=2.0)
    parser.add_argument("--top-plot", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.bundle_root.resolve()
    targets_path = (
        args.human_targets
        if args.human_targets.is_absolute()
        else root / args.human_targets
    ).resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    ).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not 0 < args.keep_fraction <= 1:
        raise ValueError("--keep-fraction must be in (0, 1]")

    progress(0, "locating inputs")
    dataset_path = args.simulation_dataset
    if dataset_path is None:
        dataset_path = find_dataset(root, smallest=args.prefer_smallest_dataset)
    elif not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    dataset_path = dataset_path.resolve()
    if not targets_path.is_file():
        raise FileNotFoundError(targets_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=False)

    progress(10, "loading human calibration targets")
    targets = validate_targets(pd.read_csv(targets_path))
    targets.to_csv(output_dir / "human_targets_used.csv", index=False)

    progress(20, f"loading simulation dataset {dataset_path.name}")
    feature_names = read_feature_names(dataset_path)
    params, stat_data = load_dataset(dataset_path)
    progress(45, f"loaded {len(params):,} simulated CPI conditions")

    groups = group_ids_from_shared_parameters(params)
    conditions = condition_table(params, stat_data, feature_names)
    conditions.insert(0, "condition_index", np.arange(len(conditions)))
    conditions.insert(1, "group_id", groups)
    add_level_scores(conditions, targets)
    progress(60, f"scoring {conditions['group_id'].nunique():,} shared profiles")

    candidates = candidate_table(conditions, targets, args.trend_weight)
    prior = proposed_prior(
        candidates,
        dataset_path,
        args.keep_fraction,
        args.minimum_keep,
    )
    progress(75, "writing ranked candidates and proposed prior")
    conditions.to_csv(output_dir / "calibration_condition_details.csv", index=False)
    candidates.to_csv(output_dir / "calibration_candidates.csv", index=False)
    (output_dir / "calibrated_parameter_ranges.yaml").write_text(
        yaml.safe_dump(prior, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    progress(85, "rendering calibration figures")
    plot_score_distribution(candidates, output_dir / "calibration_scores.png")
    plot_top_trends(
        candidates,
        conditions,
        targets,
        output_dir / "top_candidate_human_comparison.png",
        min(args.top_plot, len(candidates)),
    )

    groups_with_trends = int((candidates["n_conditions"] >= 3).sum())
    manifest = {
        "schema_version": 1,
        "created_local": datetime.now().astimezone().isoformat(),
        "source_dataset": str(dataset_path),
        "source_targets": str(targets_path),
        "condition_count": int(len(conditions)),
        "shared_profile_count": int(len(candidates)),
        "profiles_with_three_or_more_conditions": groups_with_trends,
        "hard_pass_count": int(candidates["hard_pass"].sum()),
        "best_total_score": float(candidates.iloc[0]["total_score"]),
        "trend_weight": float(args.trend_weight),
        "interpretation": (
            "If no well-scoring profile reproduces positive CPI trends, "
            "the RL policy must be recalibrated before amortizer retraining."
        ),
    }
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    progress(100, "calibration audit complete")
    print("\nTop calibration candidates:")
    display_columns = [
        "rank", "total_score", "level_score", "trend_score",
        "hard_pass", "n_conditions", *PARAMETER_NAMES[:-1],
    ]
    print(candidates[display_columns].head(10).to_string(index=False))
    print(f"\nResults: {output_dir}")
    print(f"Profiles with >=3 CPI conditions: {groups_with_trends}")
    print(f"Hard calibration passes: {int(candidates['hard_pass'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
