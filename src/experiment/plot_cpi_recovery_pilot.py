"""Plot participant-level CPI recovery in the human pilot-data style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_ROOT = ROOT / "data" / "human_experiment" / "amortizer_ready"

MODEL_PRESENTATION = {
    "PTE_CPI_CURVATURE_ONLY_24FHD_08041142": {
        "label": "Summary / curvature only",
        "filename": "cpi_recovery_summary_only.png",
    },
    "PTE_CPI_CURVATURE_24FHD_07300956": {
        "label": "With 20 Hz trajectory",
        "filename": "cpi_recovery_with_trajectory.png",
    },
}


def latest_inference(root: Path) -> Path:
    candidates = sorted(
        (path.parent for path in root.rglob("human_cpi_summary.csv")),
        key=lambda path: (path / "human_cpi_summary.csv").stat().st_mtime,
        reverse=True,
    ) if root.exists() else []
    if not candidates:
        raise FileNotFoundError(
            "No human CPI inference was found. Run python -m "
            "src.experiment.infer_human_mouse first."
        )
    return candidates[0]


def full_trial_data(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "participant_id",
        "true_cpi",
        "n_trials",
        "predicted_cpi_mean",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"CPI summary is missing columns: {sorted(missing)}")
    max_by_block = summary.groupby(
        ["model", "participant_id", "true_cpi"]
    )["n_trials"].transform("max")
    return summary[summary["n_trials"].eq(max_by_block)].copy()


def aggregate_participants(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["model", "true_cpi"], as_index=False)
        .agg(
            participant_count=("participant_id", "nunique"),
            inferred_cpi_mean=("predicted_cpi_mean", "mean"),
            inferred_cpi_sd=("predicted_cpi_mean", "std"),
            inferred_cpi_min=("predicted_cpi_mean", "min"),
            inferred_cpi_max=("predicted_cpi_mean", "max"),
        )
        .sort_values(["model", "true_cpi"])
    )


def draw_model_panel(
    axis: plt.Axes,
    data: pd.DataFrame,
    model: str,
    *,
    show_subtitle: bool,
) -> None:
    subset = data[data["model"] == model]
    if subset.empty:
        axis.set_visible(False)
        return
    participants = list(subset["participant_id"].drop_duplicates())
    colors = plt.cm.tab10(np.linspace(0.0, 0.8, len(participants)))
    for color, participant_id in zip(colors, participants):
        group = subset[subset["participant_id"] == participant_id].sort_values("true_cpi")
        axis.plot(
            group["true_cpi"],
            group["predicted_cpi_mean"],
            marker="o",
            markersize=8,
            linewidth=2,
            alpha=0.8,
            color=color,
            label=str(participant_id),
        )

    aggregate = aggregate_participants(subset)
    axis.errorbar(
        aggregate["true_cpi"],
        aggregate["inferred_cpi_mean"],
        yerr=aggregate["inferred_cpi_sd"].fillna(0.0),
        fmt="D",
        markersize=8,
        capsize=6,
        linewidth=2,
        color="0.15",
        label="mean ± SD",
        zorder=5,
    )
    lower, upper = 350.0, 1650.0
    axis.plot(
        [400, 1600],
        [400, 1600],
        linestyle="--",
        linewidth=1.5,
        color="black",
        label="identity",
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xticks([400, 800, 1600])
    axis.set_yticks([400, 800, 1600])
    axis.set_xlabel("True CPI")
    axis.set_ylabel("Inferred CPI")
    axis.grid(alpha=0.28)
    if show_subtitle:
        presentation = MODEL_PRESENTATION.get(model, {"label": model})
        axis.set_title(str(presentation["label"]), fontsize=12)

    handles, labels = axis.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = [
        *participants,
        "identity",
        "mean ± SD",
    ]
    axis.legend(
        [label_to_handle[label] for label in ordered_labels if label in label_to_handle],
        [label for label in ordered_labels if label in label_to_handle],
        loc="upper left",
        framealpha=0.9,
    )


def save_individual_plots(data: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for model in data["model"].drop_duplicates():
        presentation = MODEL_PRESENTATION.get(
            model,
            {"label": model, "filename": f"cpi_recovery_{model}.png"},
        )
        fig, axis = plt.subplots(figsize=(8, 7))
        draw_model_panel(axis, data, model, show_subtitle=False)
        fig.suptitle("CPI recovery in human pilot data", fontsize=18)
        fig.text(0.5, 0.925, str(presentation["label"]), ha="center", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        path = output_dir / str(presentation["filename"])
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def save_combined_plot(data: pd.DataFrame, output_dir: Path) -> Path:
    models = list(data["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), squeeze=False)
    for axis, model in zip(axes[0], models):
        draw_model_panel(axis, data, model, show_subtitle=True)
    fig.suptitle("CPI recovery in human pilot data", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "cpi_recovery_models_combined.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    inference_dir = args.inference_dir or latest_inference(DEFAULT_HUMAN_ROOT)
    output_dir = args.output_dir or inference_dir / "cpi_recovery_pilot_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(inference_dir / "human_cpi_summary.csv")
    data = full_trial_data(summary)
    aggregate = aggregate_participants(data)
    data.to_csv(output_dir / "cpi_recovery_participants.csv", index=False)
    aggregate.to_csv(output_dir / "cpi_recovery_mean_sd.csv", index=False)
    paths = save_individual_plots(data, output_dir)
    combined = save_combined_plot(data, output_dir)

    print(f"Participants: {', '.join(data['participant_id'].drop_duplicates())}")
    print(f"Trials per participant/CPI block: {int(data['n_trials'].max())}")
    for path in paths:
        print(f"Graph: {path.resolve()}")
    print(f"Combined graph: {combined.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
