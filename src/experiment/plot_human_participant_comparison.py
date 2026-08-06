"""Create a compact comparison of human experimental data by participant and CPI."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_ROOT = ROOT / "data" / "human_experiment" / "amortizer_ready"

METRICS = {
    "movement_time": {
        "label": "Movement time (s)",
        "scale": 1.0,
        "percent": False,
    },
    "target_hit": {
        "label": "Target hit rate",
        "scale": 1.0,
        "percent": True,
    },
    "click_distance": {
        "label": "Click error (mm)",
        "scale": 1000.0,
        "percent": False,
    },
    "max_cursor_speed": {
        "label": "Maximum cursor speed (m/s)",
        "scale": 1.0,
        "percent": False,
    },
    "cursor_total_abs_turn_rad": {
        "label": "Total trajectory turning (rad)",
        "scale": 1.0,
        "percent": False,
    },
    "cursor_path_excess_ratio": {
        "label": "Path excess ratio",
        "scale": 1.0,
        "percent": True,
    },
}


def latest_export(export_root: Path) -> Path:
    candidates = sorted(
        (
            item
            for item in export_root.iterdir()
            if item.is_dir() and (item / "human_amortizer_stats.csv").is_file()
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ) if export_root.exists() else []
    if not candidates:
        raise FileNotFoundError(
            "No human export found. Run python -m "
            "src.experiment.export_human_amortizer_data first."
        )
    return candidates[0]


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_id", "cpi", *METRICS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Human export is missing columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    for (participant_id, cpi), group in data.groupby(
        ["participant_id", "cpi"], sort=True
    ):
        row: dict[str, float | int | str] = {
            "participant_id": str(participant_id),
            "cpi": float(cpi),
            "n_trials": len(group),
        }
        for name, metadata in METRICS.items():
            values = pd.to_numeric(group[name], errors="coerce").dropna()
            scale = float(metadata["scale"])
            scaled = values * scale
            row[f"{name}_mean"] = float(scaled.mean()) if len(scaled) else np.nan
            row[f"{name}_std"] = (
                float(scaled.std(ddof=1)) if len(scaled) > 1 else np.nan
            )
            row[f"{name}_sem"] = (
                float(scaled.std(ddof=1) / np.sqrt(len(scaled)))
                if len(scaled) > 1
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["participant_id", "cpi"])


def plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0.0, 0.8, summary["participant_id"].nunique()))
    for axis, (metric, metadata) in zip(axes.flat, METRICS.items()):
        for color, (participant_id, group) in zip(
            colors, summary.groupby("participant_id", sort=True)
        ):
            group = group.sort_values("cpi")
            axis.errorbar(
                group["cpi"],
                group[f"{metric}_mean"],
                yerr=group[f"{metric}_sem"].fillna(0.0),
                marker="o",
                markersize=7,
                linewidth=2.2,
                capsize=4,
                color=color,
                label=str(participant_id),
            )
        axis.set_title(str(metadata["label"]))
        axis.set_xlabel("Physical CPI")
        axis.set_xticks(sorted(summary["cpi"].unique()))
        axis.grid(alpha=0.25)
        if bool(metadata["percent"]):
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        axis.legend()
    fig.suptitle("Human participant comparison by CPI\n(mean ± standard error)", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_dir = args.export_dir or latest_export(DEFAULT_EXPORT_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or export_dir / f"participant_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    data = pd.read_csv(export_dir / "human_amortizer_stats.csv")
    summary = summarize(data)
    summary_path = output_dir / "human_participant_comparison_summary.csv"
    plot_path = output_dir / "human_participant_comparison.png"
    summary.to_csv(summary_path, index=False)
    plot_comparison(summary, plot_path)

    print(f"Compared participants: {', '.join(summary['participant_id'].unique())}")
    print(f"Trials: {len(data)}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Graph: {plot_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
