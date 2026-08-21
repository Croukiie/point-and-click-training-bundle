"""Audit raw-count CPI cues in a v4 simulation dataset versus human data."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..config.config import INF
from ..utils.raw_count_features import (
    RAW_COUNT_BIN_HZ,
    RAW_COUNT_FEATURE_NAMES,
    RAW_COUNT_NORMALIZATION_RANGES,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_EXPORT = (
    ROOT / "data/human_experiment/amortizer_ready/rawcount125_20260821"
)
FEATURES = (*RAW_COUNT_FEATURE_NAMES, "max_cursor_speed")
TARGET_CPIS = (400.0, 800.0, 1600.0)
CONTRACT = "movement_exact20_endpoint_clickonly_rawcount125_v4"


def find_dataset() -> Path:
    candidates = list(
        (ROOT / "data/amortizer_cond/dataset").glob(f"*_{CONTRACT}")
    )
    candidates.extend(
        (ROOT / "yonsei_imports").glob(
            f"**/data/amortizer_cond/dataset/*_{CONTRACT}"
        )
    )
    candidates = [path for path in candidates if list(path.glob("vd_*.npz"))]
    if not candidates:
        raise FileNotFoundError(
            "No completed v4 validation dataset was found; pass --dataset-dir"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_simulation(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    npz_path = next(dataset_dir.glob("vd_*.npz"))
    yaml_path = next(dataset_dir.glob("vd_*.yaml"))
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=True) as archive:
        stat = archive["stat_data"]
    columns = list(metadata["stat"])
    missing = set(FEATURES) - set(columns)
    if missing:
        raise ValueError(f"Simulation dataset is missing: {sorted(missing)}")
    simulation = pd.DataFrame(
        stat.reshape(-1, stat.shape[-1]), columns=columns
    )
    condition_means = pd.DataFrame(stat.mean(axis=1), columns=columns)
    condition_means["simulated_profile"] = np.arange(len(condition_means)) // 3
    return simulation, condition_means


def per_group_slopes(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows = []
    for group, values in frame.groupby(group_column, sort=False):
        block_means = values.groupby("cpi", as_index=False)[list(FEATURES)].mean()
        if len(block_means) < 2:
            continue
        for feature in FEATURES:
            rows.append({
                "group": group,
                "feature": feature,
                "slope": float(
                    np.polyfit(block_means["cpi"], block_means[feature], 1)[0]
                ),
            })
    return pd.DataFrame(rows)


def normalization_range(feature: str) -> tuple[float, float]:
    if feature in RAW_COUNT_NORMALIZATION_RANGES:
        configured = RAW_COUNT_NORMALIZATION_RANGES[feature]
    else:
        configured = INF.normalize.curvature.stat.range[feature]
    return float(configured["min"]), float(configured["max"])


def run_audit(
    human_export: Path,
    dataset_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    simulation, conditions = load_simulation(dataset_dir)
    human = pd.read_csv(human_export / "human_amortizer_stats.csv")
    missing = set(FEATURES) - set(human.columns)
    if missing:
        raise ValueError(
            f"Human export is missing {sorted(missing)}; re-export with the v2 exporter"
        )

    sim_slopes = per_group_slopes(conditions, "simulated_profile")
    human_slopes = per_group_slopes(human, "participant_id")
    rows = []
    for feature in FEATURES:
        sim_feature_slopes = sim_slopes.loc[
            sim_slopes["feature"] == feature, "slope"
        ]
        human_feature_slopes = human_slopes.loc[
            human_slopes["feature"] == feature, "slope"
        ]
        sim_slope = float(sim_feature_slopes.mean())
        human_slope = float(human_feature_slopes.mean())
        sim_min = float(simulation[feature].min())
        sim_max = float(simulation[feature].max())
        norm_min, norm_max = normalization_range(feature)
        outside_sim = float(
            100
            * np.mean(
                (human[feature].to_numpy(dtype=float) < sim_min)
                | (human[feature].to_numpy(dtype=float) > sim_max)
            )
        )

        for cpi in TARGET_CPIS:
            sim_at_cpi = simulation[np.isclose(simulation["cpi"], cpi)][feature]
            human_at_cpi = human[np.isclose(human["cpi"], cpi)][feature]
            rows.append({
                "feature": feature,
                "cpi": int(cpi),
                "simulation_n": int(len(sim_at_cpi)),
                "simulation_raw_mean": (
                    float(sim_at_cpi.mean()) if len(sim_at_cpi) else np.nan
                ),
                "simulation_raw_sd": (
                    float(sim_at_cpi.std(ddof=1)) if len(sim_at_cpi) > 1 else np.nan
                ),
                "human_n": int(len(human_at_cpi)),
                "human_raw_mean": float(human_at_cpi.mean()),
                "human_raw_sd": float(human_at_cpi.std(ddof=1)),
                "simulation_within_profile_slope_mean": sim_slope,
                "simulation_within_profile_slope_sd": float(
                    sim_feature_slopes.std(ddof=1)
                ),
                "human_within_participant_slope_mean": human_slope,
                "human_within_participant_slope_sd": float(
                    human_feature_slopes.std(ddof=1)
                ),
                "slope_direction_agrees": bool(
                    np.sign(sim_slope) == np.sign(human_slope)
                ),
                "human_outside_simulated_range_pct": outside_sim,
                "simulation_raw_min": sim_min,
                "simulation_raw_max": sim_max,
                "normalization_min": norm_min,
                "normalization_max": norm_max,
                "human_outside_normalization_pct": float(
                    100
                    * np.mean(
                        (human[feature] < norm_min) | (human[feature] > norm_max)
                    )
                ),
            })

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    result.to_csv(output_dir / "raw_count_feature_audit.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    for axis, feature in zip(axes.flat, FEATURES):
        sim_trend = simulation.groupby("cpi", as_index=False)[feature].mean()
        axis.plot(
            sim_trend["cpi"], sim_trend[feature],
            color="#2878b5", alpha=0.75, label="simulation",
        )
        for participant, values in human.groupby("participant_id", sort=True):
            trend = values.groupby("cpi", as_index=False)[feature].mean()
            axis.plot(
                trend["cpi"], trend[feature], marker="o", linewidth=2,
                label=str(participant),
            )
        axis.set_title(feature)
        axis.set_xlabel("CPI")
        axis.set_ylabel("raw value")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(2, len(labels)))
    fig.suptitle("Raw-count features: simulation versus human CPI trends")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(
        output_dir / "raw_count_feature_trends.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    feature_summary = result.drop_duplicates("feature")
    opposite = feature_summary.loc[
        ~feature_summary["slope_direction_agrees"], "feature"
    ].tolist()
    missing_exact = result.loc[
        result["simulation_n"] == 0, ["feature", "cpi"]
    ].to_dict("records")
    summary = {
        "question": (
            "Do raw-count CPI cues learned in simulation exist with the same "
            "direction in human behavior?"
        ),
        "dataset_contract": CONTRACT,
        "human_export": str(human_export.resolve()),
        "simulation_dataset": str(dataset_dir.resolve()),
        "raw_count_bin_hz": RAW_COUNT_BIN_HZ,
        "units": "integer sensor counts and seconds; no CPI-derived metres",
        "reaction_interval": "removed before exact fixed-rate count binning",
        "terminal_sample": "count packets through the click timestamp are retained",
        "simulation_slopes": "calculated within shared profiles",
        "human_slopes": "calculated within participant",
        "opposite_or_missing_human_trends": opposite,
        "missing_exact_simulation_cpi_rows": missing_exact,
    }
    (output_dir / "raw_count_feature_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-export", type=Path, default=DEFAULT_HUMAN_EXPORT)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or (
        ROOT
        / "data/diagnostics"
        / f"raw_count_feature_audit_{datetime.now():%Y%m%d_%H%M%S}"
    )
    result = run_audit(
        args.human_export,
        args.dataset_dir or find_dataset(),
        output_dir,
    )
    print(result.to_string(index=False))
    print(f"OUTPUT_DIR={output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
