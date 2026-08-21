"""Audit summary-feature domain gaps for the final integrity CPI models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "data/human_experiment/amortizer_ready/true20hz_20260811_121920"
DEFAULT_MODEL = (
    ROOT
    / "data/amortizer_cond/models"
    / "PTE_SHARED_CPI_INTEGRITY_CORE3_FULL_integrity_visible_260819_151300_SUMMARY_08210246"
)
FEATURES = (
    "movement_time",
    "target_hit",
    "click_distance",
    "target_rel_pos_x0",
    "target_rel_pos_y0",
    "click_rel_endpoint_x",
    "click_rel_endpoint_y",
    "target_radius",
    "target_distance",
    "traverse_length",
    "max_cursor_speed",
    "cursor_turn_energy_rad2",
    "cursor_early_mean_abs_turn_rad",
    "cursor_late_mean_abs_turn_rad",
)


def latest_validation_dataset() -> Path:
    imports = sorted(
        (ROOT / "yonsei_imports").glob("integrity_visible_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for imported in imports:
        parent = imported / "data/amortizer_cond/dataset"
        if not parent.exists():
            continue
        directories = list(parent.iterdir())
        if directories:
            return directories[0]
    raise FileNotFoundError("No imported integrity validation dataset was found")


def long_path(path: Path) -> Path:
    text = str(path.resolve())
    return Path("\\\\?\\" + text) if len(text) >= 240 else path


def slopes(frame: pd.DataFrame, group: str, features: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for label, subset in frame.groupby(group, sort=False):
        block_means = subset.groupby("cpi", as_index=False)[list(features)].mean()
        if len(block_means) < 2:
            continue
        for feature in features:
            slope = np.polyfit(block_means["cpi"], block_means[feature], 1)[0]
            rows.append({"group": label, "feature": feature, "slope": float(slope)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    dataset_dir = long_path(args.dataset_dir or latest_validation_dataset())
    npz_path = next(dataset_dir.glob("vd_*.npz"))
    yaml_path = next(dataset_dir.glob("vd_*.yaml"))
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=True) as archive:
        stat = archive["stat_data"]
    columns = metadata["stat"]
    simulation = pd.DataFrame(stat.reshape(-1, stat.shape[-1]), columns=columns)
    condition_means = pd.DataFrame(stat.mean(axis=1), columns=columns)
    condition_means["simulated_profile"] = np.arange(len(condition_means)) // 3

    human = pd.read_csv(args.human_export / "human_amortizer_stats.csv")
    sim_slopes = slopes(condition_means, "simulated_profile", FEATURES)
    human_slopes = slopes(human, "participant_id", FEATURES)
    normalization = yaml.safe_load(
        (DEFAULT_MODEL / "data_normalization_setting.yaml").read_text(encoding="utf-8")
    )

    rows = []
    for feature in FEATURES:
        sim_values = simulation[feature].to_numpy(dtype=float)
        human_values = human[feature].to_numpy(dtype=float)
        sim_feature_slopes = sim_slopes.loc[sim_slopes.feature == feature, "slope"]
        human_feature_slopes = human_slopes.loc[human_slopes.feature == feature]
        participant_slopes = dict(zip(human_feature_slopes["group"], human_feature_slopes["slope"]))
        sim_slope = float(sim_feature_slopes.mean())
        human_slope = float(human_feature_slopes["slope"].mean())
        configured = normalization["stat"]["range"][feature]
        sim_min = float(np.min(sim_values))
        sim_max = float(np.max(sim_values))
        # Treat values equal to a simulated boundary as in-range.  This matters
        # for exact protocol constants such as the 1.25 and 7 mm target radii,
        # whose CSV/NPZ representations can differ by a few floating-point ulps.
        sim_tolerance = max(1e-12, (sim_max - sim_min) * 1e-9)
        rows.append(
            {
                "feature": feature,
                "simulation_mean": float(np.mean(sim_values)),
                "simulation_sd": float(np.std(sim_values, ddof=1)),
                "simulation_min": sim_min,
                "simulation_max": sim_max,
                "human_mean": float(np.mean(human_values)),
                "human_sd": float(np.std(human_values, ddof=1)),
                "human_min": float(np.min(human_values)),
                "human_max": float(np.max(human_values)),
                "human_outside_simulation_range_pct": float(
                    100
                    * np.mean(
                        (human_values < sim_min - sim_tolerance)
                        | (human_values > sim_max + sim_tolerance)
                    )
                ),
                "simulation_within_profile_slope_mean": sim_slope,
                "simulation_within_profile_slope_sd": float(sim_feature_slopes.std(ddof=1)),
                "human_participant_slope_mean": human_slope,
                "human_slope_elisa": float(participant_slopes.get("Elisa", np.nan)),
                "human_slope_hyeeun": float(participant_slopes.get("Hyeeun", np.nan)),
                "mean_slope_direction_agrees": bool(np.sign(sim_slope) == np.sign(human_slope)),
                "normalization_min": float(configured["min"]),
                "normalization_max": float(configured["max"]),
                "human_outside_normalization_range_pct": float(
                    100
                    * np.mean(
                        (human_values < float(configured["min"]))
                        | (human_values > float(configured["max"]))
                    )
                ),
            }
        )

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "feature_domain_audit.csv", index=False)

    focus = [
        "target_hit",
        "movement_time",
        "max_cursor_speed",
        "traverse_length",
        "cursor_turn_energy_rad2",
        "cursor_early_mean_abs_turn_rad",
        "cursor_late_mean_abs_turn_rad",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for axis, feature in zip(axes.flat, focus):
        sim_group = condition_means.groupby("cpi")[feature].mean().reset_index()
        axis.plot(sim_group.cpi, sim_group[feature], color="#2878b5", label="simulation")
        for participant, group in human.groupby("participant_id"):
            human_group = group.groupby("cpi")[feature].mean().reset_index()
            axis.plot(human_group.cpi, human_group[feature], marker="o", label=participant)
        axis.set_title(feature)
        axis.set_xlabel("CPI")
        axis.set_ylabel("raw value")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(focus):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Integrity simulation versus human v2 feature trends")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(args.output_dir / "feature_domain_trends.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
