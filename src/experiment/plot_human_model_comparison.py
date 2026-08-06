"""Compare human experiment metrics with the RL simulator population."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import yaml

from .plot_human_participant_comparison import (
    DEFAULT_EXPORT_ROOT,
    METRICS,
    latest_export,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DATASET = (
    ROOT
    / "data"
    / "amortizer"
    / "dataset"
    / "pnc_rl_24fhd_cpi_26210DBCT_best_20_default"
)
DEFAULT_CPIS = (400, 800, 1600)


def largest_validation_dataset(dataset_dir: Path) -> Path:
    candidates: list[tuple[int, int, Path]] = []
    pattern = re.compile(r"vd_(\d+)users_(\d+)trials_.*\.npz$")
    for path in dataset_dir.glob("vd_*users_*trials_*.npz"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), int(match.group(2)), path))
    if not candidates:
        raise FileNotFoundError(f"No validation simulation was found in {dataset_dir}")
    return max(candidates, key=lambda item: (item[0], item[1], item[2].stat().st_mtime))[2]


def human_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (participant_id, cpi), group in data.groupby(
        ["participant_id", "cpi"], sort=True
    ):
        for metric, metadata in METRICS.items():
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            values = values * float(metadata["scale"])
            rows.append({
                "source_type": "human",
                "source_name": str(participant_id),
                "cpi": float(cpi),
                "metric": metric,
                "metric_label": str(metadata["label"]),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "sem": (
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else np.nan
                ),
                "n_independent_units": len(values),
                "n_trials": len(values),
                "model_cpi_min": np.nan,
                "model_cpi_max": np.nan,
            })
    return pd.DataFrame(rows)


def model_summary(
    dataset_path: Path,
    cpis: tuple[int, ...],
    cpi_bandwidth: float,
) -> pd.DataFrame:
    feature_info = yaml.safe_load(dataset_path.with_suffix(".yaml").read_text(encoding="utf-8"))
    feature_names = list(feature_info["stat"])
    required = {"cpi", *METRICS}
    missing = required - set(feature_names)
    if missing:
        raise ValueError(f"Simulation dataset is missing columns: {sorted(missing)}")

    with np.load(dataset_path, allow_pickle=False) as archive:
        stats = archive["stat_data"]
    cpi_index = feature_names.index("cpi")
    user_cpis = stats[:, 0, cpi_index].astype(float)
    rows: list[dict[str, float | int | str]] = []

    for target_cpi in cpis:
        selected = np.flatnonzero(np.abs(user_cpis - target_cpi) <= cpi_bandwidth)
        if not len(selected):
            selected = np.array([int(np.argmin(np.abs(user_cpis - target_cpi)))])
        for metric, metadata in METRICS.items():
            metric_index = feature_names.index(metric)
            # Trials from one simulated user share latent parameters. Average
            # within user first, then calculate uncertainty across users.
            per_user = np.nanmean(stats[selected, :, metric_index], axis=1)
            per_user = per_user * float(metadata["scale"])
            rows.append({
                "source_type": "model",
                "source_name": "RL simulator population",
                "cpi": float(target_cpi),
                "metric": metric,
                "metric_label": str(metadata["label"]),
                "mean": float(np.mean(per_user)),
                "std": float(np.std(per_user, ddof=1)) if len(per_user) > 1 else np.nan,
                "sem": (
                    float(np.std(per_user, ddof=1) / np.sqrt(len(per_user)))
                    if len(per_user) > 1
                    else np.nan
                ),
                "n_independent_units": len(per_user),
                "n_trials": int(len(per_user) * stats.shape[1]),
                "model_cpi_min": float(np.min(user_cpis[selected])),
                "model_cpi_max": float(np.max(user_cpis[selected])),
            })
    return pd.DataFrame(rows)


def comparison_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    human = summary[summary["source_type"] == "human"]
    model = summary[summary["source_type"] == "model"][
        ["cpi", "metric", "mean"]
    ].rename(columns={"mean": "model_mean"})
    gaps = human.merge(model, on=["cpi", "metric"], how="left")
    gaps = gaps.rename(columns={"mean": "human_mean", "source_name": "participant_id"})
    gaps["human_minus_model"] = gaps["human_mean"] - gaps["model_mean"]
    gaps["relative_gap"] = np.where(
        np.abs(gaps["model_mean"]) > 1e-12,
        gaps["human_minus_model"] / gaps["model_mean"],
        np.nan,
    )
    return gaps[
        [
            "participant_id",
            "cpi",
            "metric",
            "metric_label",
            "human_mean",
            "model_mean",
            "human_minus_model",
            "relative_gap",
        ]
    ].sort_values(["metric", "participant_id", "cpi"])


def plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    humans = summary[summary["source_type"] == "human"]
    model = summary[summary["source_type"] == "model"]
    participants = list(humans["source_name"].drop_duplicates())
    colors = plt.cm.tab10(np.linspace(0.0, 0.8, len(participants)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)

    for axis, (metric, metadata) in zip(axes.flat, METRICS.items()):
        for color, participant in zip(colors, participants):
            group = humans[
                (humans["metric"] == metric) & (humans["source_name"] == participant)
            ].sort_values("cpi")
            axis.errorbar(
                group["cpi"],
                group["mean"],
                yerr=group["sem"].fillna(0.0),
                marker="o",
                linewidth=2.2,
                capsize=4,
                color=color,
                label=participant,
            )
        model_group = model[model["metric"] == metric].sort_values("cpi")
        axis.errorbar(
            model_group["cpi"],
            model_group["mean"],
            yerr=model_group["sem"].fillna(0.0),
            marker="s",
            markersize=7,
            linestyle="--",
            linewidth=2.4,
            capsize=4,
            color="black",
            label="RL model population",
        )
        axis.set_title(str(metadata["label"]))
        axis.set_xlabel("Physical CPI")
        axis.set_xticks(sorted(summary["cpi"].unique()))
        axis.grid(alpha=0.25)
        if bool(metadata["percent"]):
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        if metric == "click_distance":
            axis.set_yscale("log")
            axis.set_ylabel("Logarithmic scale")
        axis.legend(fontsize=8)

    fig.suptitle(
        "Human data compared with the RL simulator population\n"
        "(mean ± standard error; model uses nearby CPI users)",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--model-dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cpi-bandwidth", type=float, default=100.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_dir = args.export_dir or latest_export(DEFAULT_EXPORT_ROOT)
    dataset_path = args.model_dataset or largest_validation_dataset(DEFAULT_MODEL_DATASET)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or export_dir / f"human_model_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    human_data = pd.read_csv(export_dir / "human_amortizer_stats.csv")
    summary = pd.concat(
        [
            human_summary(human_data),
            model_summary(dataset_path, DEFAULT_CPIS, args.cpi_bandwidth),
        ],
        ignore_index=True,
    )
    gaps = comparison_gaps(summary)
    summary_path = output_dir / "human_model_comparison_summary.csv"
    gap_path = output_dir / "human_model_gaps.csv"
    plot_path = output_dir / "human_model_comparison.png"
    summary.to_csv(summary_path, index=False)
    gaps.to_csv(gap_path, index=False)
    plot_comparison(summary, plot_path)

    model_counts = summary[summary["source_type"] == "model"][
        ["cpi", "n_independent_units", "model_cpi_min", "model_cpi_max"]
    ].drop_duplicates()
    print(f"Human trials: {len(human_data)}")
    print(f"Simulation dataset: {dataset_path.resolve()}")
    print("Model users selected by CPI:")
    print(model_counts.to_string(index=False))
    print(f"Summary: {summary_path.resolve()}")
    print(f"Gaps: {gap_path.resolve()}")
    print(f"Graph: {plot_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
