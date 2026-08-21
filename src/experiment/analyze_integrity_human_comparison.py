"""Compare final integrity-chain shared-CPI amortizers with human pilot data."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .infer_human_shared_cpi import shared_infer_subset, trial_subset_hash
from ..inference.pnc_inferer_2step import PnCConditionalInferer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = (
    ROOT / "data/human_experiment/amortizer_ready/true20hz_20260811_121920"
)
DEFAULT_MODELS = {
    "speedpath": (
        "PTE_SHARED_CPI_INTEGRITY_SPEEDPATH_FULL_"
        "integrity_visible_260819_151300_SUMMARY_08202351"
    ),
    "core3": (
        "PTE_SHARED_CPI_INTEGRITY_CORE3_FULL_"
        "integrity_visible_260819_151300_SUMMARY_08210246"
    ),
}


def predictive_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - true
    sst = float(np.sum((true - np.mean(true)) ** 2))
    slope, intercept = np.polyfit(true, predicted, 1)
    return {
        "n_predictions": int(len(true)),
        "predictive_r2": float(1.0 - np.sum(residual**2) / sst) if sst else np.nan,
        "pearson_r": float(np.corrcoef(true, predicted)[0, 1])
        if np.std(true) and np.std(predicted)
        else np.nan,
        "slope": float(slope),
        "intercept": float(intercept),
        "mae_cpi": float(np.mean(np.abs(residual))),
        "rmse_cpi": float(np.sqrt(np.mean(residual**2))),
        "bias_cpi": float(np.mean(residual)),
    }


def ordering_metrics(frame: pd.DataFrame, participant_column: str) -> dict[str, float]:
    exact = []
    pairwise = []
    spearman = []
    for _, group in frame.groupby(participant_column, sort=False):
        ordered = group.sort_values("true_cpi")
        inferred = ordered["predicted_cpi"].to_numpy(dtype=float)
        exact.append(float(np.all(np.diff(inferred) > 0)))
        comparisons = [
            float(inferred[left] < inferred[right])
            for left in range(len(inferred))
            for right in range(left + 1, len(inferred))
        ]
        pairwise.append(float(np.mean(comparisons)))
        ranks = np.argsort(np.argsort(inferred))
        spearman.append(float(np.corrcoef(np.arange(len(inferred)), ranks)[0, 1]))
    return {
        "exact_order_accuracy": float(np.mean(exact)),
        "pairwise_order_accuracy": float(np.mean(pairwise)),
        "mean_spearman_order": float(np.mean(spearman)),
    }


def simulated_results(models: dict[str, str], trial_counts: list[int]):
    rows = []
    metrics = []
    for label, model in models.items():
        model_root = ROOT / "data/amortizer_cond/models" / model
        result_root = model_root / "results/iter199"
        for count in trial_counts:
            source = result_root / f"trial{count:03d}_cpi_ordering.csv"
            frame = pd.read_csv(source).rename(columns={"inferred_cpi": "predicted_cpi"})
            frame["model_variant"] = label
            frame["n_trials_per_block"] = count
            rows.append(frame)
            metric = {
                "source": "simulation",
                "model_variant": label,
                "n_trials_per_block": count,
                **predictive_metrics(frame["true_cpi"], frame["predicted_cpi"]),
                **ordering_metrics(frame, "simulated_participant"),
            }
            metrics.append(metric)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def human_predictions(
    export_dir: Path,
    models: dict[str, str],
    trial_counts: list[int],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    stats = pd.read_csv(export_dir / "human_amortizer_stats.csv")
    inferers = {
        label: PnCConditionalInferer(model, ckpt=200, verbose=True)
        for label, model in models.items()
    }
    rng = np.random.default_rng(seed)
    rows = []

    for (session_id, participant_id), participant in stats.groupby(
        ["session_id", "participant_id"], sort=True
    ):
        blocks = list(participant.groupby("block_index", sort=True))
        if len(blocks) != 3:
            raise ValueError(f"{participant_id}/{session_id}: expected three CPI blocks")
        minimum = min(len(block) for _, block in blocks)
        for count in trial_counts:
            if count > minimum:
                continue
            for repeat in range(repeats):
                selected_blocks = []
                for anonymous_block, (_, block) in enumerate(blocks):
                    block = block.reset_index(drop=True)
                    indices = np.sort(rng.choice(len(block), size=count, replace=False))
                    selected = block.iloc[indices].copy().reset_index(drop=True)
                    selected["_anonymous_block"] = anonymous_block
                    selected_blocks.append(selected)
                joint = pd.concat(selected_blocks, ignore_index=True)
                subset_sha = trial_subset_hash(joint)

                for label, inferer in inferers.items():
                    shared, block_specific = shared_infer_subset(inferer, joint, None)
                    for anonymous_block, selected in enumerate(selected_blocks):
                        true_cpi = float(selected["cpi"].iloc[0])
                        predicted = float(block_specific[anonymous_block]["cpi"])
                        rows.append(
                            {
                                "source": "human",
                                "model_variant": label,
                                "model": models[label],
                                "session_id": session_id,
                                "participant_id": participant_id,
                                "anonymous_block": anonymous_block,
                                "true_cpi": true_cpi,
                                "n_trials_per_block": count,
                                "repeat": repeat,
                                "trial_subset_sha256": subset_sha,
                                "predicted_cpi": predicted,
                                "signed_error": predicted - true_cpi,
                                "absolute_error": abs(predicted - true_cpi),
                                **shared,
                            }
                        )
    return pd.DataFrame(rows)


def human_summaries(predictions: pd.DataFrame):
    summary = (
        predictions.groupby(
            [
                "model_variant",
                "model",
                "participant_id",
                "session_id",
                "true_cpi",
                "n_trials_per_block",
            ],
            as_index=False,
        )
        .agg(
            repeats=("repeat", "count"),
            predicted_cpi_mean=("predicted_cpi", "mean"),
            predicted_cpi_sd=("predicted_cpi", "std"),
            mae_cpi=("absolute_error", "mean"),
            bias_cpi=("signed_error", "mean"),
        )
        .sort_values(["model_variant", "participant_id", "n_trials_per_block", "true_cpi"])
    )

    metric_rows = []
    ordering_rows = []
    for (label, count), group in predictions.groupby(
        ["model_variant", "n_trials_per_block"], sort=True
    ):
        metric_rows.append(
            {
                "source": "human",
                "model_variant": label,
                "n_trials_per_block": int(count),
                **predictive_metrics(group["true_cpi"], group["predicted_cpi"]),
            }
        )
        per_repeat = []
        for (participant, repeat), subset in group.groupby(
            ["participant_id", "repeat"], sort=False
        ):
            subset = subset.copy()
            subset["participant_repeat"] = f"{participant}:{repeat}"
            per_repeat.append(subset)
        order = ordering_metrics(pd.concat(per_repeat), "participant_repeat")
        metric_rows[-1].update(order)

    for keys, group in summary.groupby(
        ["model_variant", "participant_id", "n_trials_per_block"], sort=True
    ):
        ordered = group.sort_values("true_cpi")
        values = ordered["predicted_cpi_mean"].to_numpy(dtype=float)
        ordering_rows.append(
            {
                "model_variant": keys[0],
                "participant_id": keys[1],
                "n_trials_per_block": int(keys[2]),
                "cpi_400": float(values[0]),
                "cpi_800": float(values[1]),
                "cpi_1600": float(values[2]),
                "strictly_increasing": bool(np.all(np.diff(values) > 0)),
                "slope": float(
                    np.polyfit(ordered["true_cpi"].to_numpy(dtype=float), values, 1)[0]
                ),
            }
        )
    return summary, pd.DataFrame(metric_rows), pd.DataFrame(ordering_rows)


def plot_human(summary: pd.DataFrame, output: Path, trial_count: int):
    selected = summary[summary["n_trials_per_block"] == trial_count]
    participants = sorted(selected["participant_id"].unique())
    variants = [label for label in ("speedpath", "core3") if label in set(selected["model_variant"])]
    fig, axes = plt.subplots(
        len(variants), len(participants), figsize=(6 * len(participants), 4.5 * len(variants)), squeeze=False
    )
    colors = {"speedpath": "#2878b5", "core3": "#d65f5f"}
    for row, variant in enumerate(variants):
        for column, participant in enumerate(participants):
            axis = axes[row, column]
            group = selected[
                (selected["model_variant"] == variant)
                & (selected["participant_id"] == participant)
            ].sort_values("true_cpi")
            axis.errorbar(
                group["true_cpi"], group["predicted_cpi_mean"],
                yerr=group["predicted_cpi_sd"].fillna(0), marker="o", capsize=4,
                linewidth=2.2, color=colors[variant], label=variant,
            )
            axis.plot([400, 1600], [400, 1600], "--", color="0.25", label="identity")
            axis.set_title(f"{participant} — {variant}")
            axis.set_xlabel("True CPI")
            axis.set_ylabel("Inferred CPI")
            axis.grid(alpha=0.25)
            axis.legend()
    fig.suptitle(f"Integrity shared-CPI amortizers on human pilot data ({trial_count} trials/block)")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trial-counts", default="16,32,64")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args(argv)

    trial_counts = sorted({int(value) for value in args.trial_counts.split(",")})
    output_dir = args.output_dir or (
        args.export_dir / f"integrity_visible_comparison_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    predictions = human_predictions(
        args.export_dir, DEFAULT_MODELS, trial_counts, args.repeats, args.seed
    )
    human_summary, human_metrics, human_ordering = human_summaries(predictions)
    simulated_predictions, simulated_metrics = simulated_results(DEFAULT_MODELS, trial_counts)
    combined_metrics = pd.concat([simulated_metrics, human_metrics], ignore_index=True)

    predictions.to_csv(output_dir / "human_predictions.csv", index=False)
    human_summary.to_csv(output_dir / "human_cpi_summary.csv", index=False)
    human_ordering.to_csv(output_dir / "human_cpi_ordering.csv", index=False)
    simulated_predictions.to_csv(output_dir / "simulated_predictions.csv", index=False)
    combined_metrics.to_csv(output_dir / "simulation_vs_human_metrics.csv", index=False)
    plot_human(human_summary, output_dir / "human_cpi_comparison.png", max(trial_counts))

    stats = pd.read_csv(args.export_dir / "human_amortizer_stats.csv")
    manifest = {
        "created_local": datetime.now().astimezone().isoformat(),
        "models": DEFAULT_MODELS,
        "checkpoint": 200,
        "human_export": str(args.export_dir.resolve()),
        "human_protocol": "target_radius_matched_v2",
        "simulation_contract": "movement_exact20_endpoint_clickonly_v3",
        "interpretation": "diagnostic_only_until_v3_human_trials_are_collected",
        "trial_counts": trial_counts,
        "bootstrap_repeats": args.repeats,
        "seed": args.seed,
        "human_participants": sorted(stats["participant_id"].unique().tolist()),
        "human_trial_count": int(len(stats)),
        "human_target_radius_range_m": [
            float(stats["target_radius"].min()), float(stats["target_radius"].max())
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(combined_metrics.to_string(index=False))
    print(f"OUTPUT_DIR={output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
