"""Compare independent and shared-parameter CPI inference on identical trials."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..inference.pnc_inferer import PnCInferer
from ..inference.pnc_inferer_2step import PnCConditionalInferer
from ..scripts.shared_cpi_2step_run import BLOCK_PARAMETERS, SHARED_PARAMETERS
from .infer_human_cpi import (
    DEFAULT_HUMAN_ROOT,
    infer_subset,
    latest_export,
    model_uses_trajectory,
    trajectory_lookup,
)


DEFAULT_INDEPENDENT_TRAJECTORY = "PTE_CPI_CURVATURE_24FHD_07300956"
DEFAULT_INDEPENDENT_SUMMARY = "PTE_CPI_CURVATURE_ONLY_24FHD_08041142"


def trial_subset_hash(stats: pd.DataFrame) -> str:
    payload = "\n".join(sorted(stats["trial_uid"].astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def shared_infer_subset(
    inferer: PnCConditionalInferer,
    stats: pd.DataFrame,
    trajectories: dict[str, pd.DataFrame] | None,
) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    trial_trajectories = None
    if model_uses_trajectory(inferer):
        if trajectories is None:
            raise ValueError("This shared amortizer requires trajectory data")
        trial_trajectories = []
        for uid in stats["trial_uid"].astype(str):
            if uid not in trajectories:
                raise ValueError(f"Missing trajectory for {uid}")
            trial_trajectories.append(trajectories[uid])
    normalized_stats, normalized_trajectories, grouping = inferer.process_data(
        stats,
        trial_trajectories,
        grouping_index_column="_anonymous_block",
    )
    _, _, shared_parameters, block_parameters = inferer.infer(
        normalized_stats,
        normalized_trajectories,
        grouping_index=np.asarray(grouping, dtype=np.int64),
        type="mode",
    )
    return shared_parameters, block_parameters


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "fit_structure",
        "model",
        "uses_trajectory",
        "participant_id",
        "session_id",
        "true_cpi",
        "n_trials_per_block",
    ]
    return (
        predictions.groupby(grouping, as_index=False)
        .agg(
            repeats=("repeat", "count"),
            predicted_cpi_mean=("predicted_cpi", "mean"),
            predicted_cpi_std=("predicted_cpi", "std"),
            signed_error_mean=("signed_error", "mean"),
            absolute_error_mean=("absolute_error", "mean"),
        )
        .sort_values(
            ["fit_structure", "participant_id", "n_trials_per_block", "true_cpi"]
        )
        .reset_index(drop=True)
    )


def order_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = [
        "fit_structure",
        "model",
        "participant_id",
        "session_id",
        "n_trials_per_block",
    ]
    for keys, group in summary.groupby(grouping, sort=False):
        ordered = group.sort_values("true_cpi")
        true_values = ordered["true_cpi"].to_numpy(dtype=float)
        inferred = ordered["predicted_cpi_mean"].to_numpy(dtype=float)
        rows.append({
            **dict(zip(grouping, keys)),
            "true_cpi_order": ";".join(f"{value:g}" for value in true_values),
            "inferred_cpi_in_true_order": ";".join(
                f"{value:.2f}" for value in inferred
            ),
            "strictly_increasing": bool(np.all(np.diff(inferred) > 0)),
            "spearman_order": (
                float(
                    np.corrcoef(
                        np.argsort(np.argsort(true_values)),
                        np.argsort(np.argsort(inferred)),
                    )[0, 1]
                )
                if len(inferred) >= 2 and np.std(inferred) > 0
                else np.nan
            ),
        })
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    participants = list(summary["participant_id"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(participants),
        figsize=(6.5 * len(participants), 5),
        squeeze=False,
    )
    colors = {"independent": "#e67e22", "shared": "#2878b5"}
    for axis, participant in zip(axes[0], participants):
        subset = summary[summary["participant_id"] == participant]
        for structure, group in subset.groupby("fit_structure", sort=False):
            group = group.sort_values("true_cpi")
            axis.errorbar(
                group["true_cpi"],
                group["predicted_cpi_mean"],
                yerr=group["predicted_cpi_std"].fillna(0.0),
                marker="o",
                linewidth=2.2,
                capsize=3,
                color=colors[structure],
                label=structure,
            )
        axis.plot([400, 1600], [400, 1600], "--", color="0.25", label="identity")
        axis.set_title(str(participant))
        axis.set_xlabel("True CPI (evaluation only)")
        axis.set_ylabel("Inferred CPI")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Independent versus shared-parameter CPI inference")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_comparison(
    export_dir: Path,
    output_dir: Path,
    *,
    shared_model: str,
    shared_checkpoint: int | None,
    independent_model: str | None,
    independent_checkpoint: int,
    trial_counts: list[int] | None,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    stats = pd.read_csv(export_dir / "human_amortizer_stats.csv")
    required = {
        "trial_uid",
        "session_id",
        "participant_id",
        "block_index",
        "cpi",
    }
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"Statistics export is missing columns: {sorted(missing)}")

    shared = PnCConditionalInferer(
        shared_model,
        ckpt=shared_checkpoint,
        verbose=True,
    )
    if tuple(shared.simulator.main_param_list) != SHARED_PARAMETERS:
        raise ValueError(
            "The selected 2-step model does not contain the expected nine "
            "shared parameters"
        )
    if tuple(shared.simulator.sub_param_list) != BLOCK_PARAMETERS:
        raise ValueError("The selected 2-step model must use CPI as its only sub parameter")

    shared_uses_trajectory = model_uses_trajectory(shared)
    if independent_model is None:
        independent_model = (
            DEFAULT_INDEPENDENT_TRAJECTORY
            if shared_uses_trajectory
            else DEFAULT_INDEPENDENT_SUMMARY
        )
    independent = PnCInferer(
        independent_model,
        ckpt=independent_checkpoint,
        verbose=True,
    )
    if model_uses_trajectory(independent) != shared_uses_trajectory:
        raise ValueError(
            "Shared and independent amortizers must use the same trajectory features"
        )

    trajectories = (
        trajectory_lookup(export_dir / "human_amortizer_trajectories.csv")
        if shared_uses_trajectory
        else None
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    subset_manifest: list[dict[str, Any]] = []

    for (session_id, participant_id), participant in stats.groupby(
        ["session_id", "participant_id"], sort=True
    ):
        block_items = list(participant.groupby("block_index", sort=True))
        if len(block_items) != 3:
            raise ValueError(
                f"{participant_id}/{session_id} has {len(block_items)} blocks; expected 3"
            )
        minimum_trials = min(len(group) for _, group in block_items)
        participant_counts = trial_counts or [minimum_trials]

        for n_trials in participant_counts:
            if n_trials > minimum_trials:
                continue
            repeats = 1 if n_trials == minimum_trials else bootstrap_repeats
            for repeat in range(repeats):
                selected_blocks: list[tuple[int, float, pd.DataFrame]] = []
                for anonymous_group, (block_label, block) in enumerate(block_items):
                    block = block.reset_index(drop=True)
                    if n_trials == len(block):
                        selected = block.copy()
                    else:
                        indices = np.sort(
                            rng.choice(len(block), size=n_trials, replace=False)
                        )
                        selected = block.iloc[indices].reset_index(drop=True)
                    true_values = selected["cpi"].drop_duplicates()
                    if len(true_values) != 1:
                        raise ValueError(f"Block {block_label} contains multiple CPI values")
                    true_cpi = float(true_values.iloc[0])
                    selected["_anonymous_block"] = anonymous_group
                    selected_blocks.append((anonymous_group, true_cpi, selected))
                    subset_manifest.append({
                        "session_id": session_id,
                        "participant_id": participant_id,
                        "block_label": block_label,
                        "anonymous_group": anonymous_group,
                        "true_cpi_evaluation_only": true_cpi,
                        "n_trials": n_trials,
                        "repeat": repeat,
                        "trial_subset_sha256": trial_subset_hash(selected),
                    })

                joint_stats = pd.concat(
                    [item[2] for item in selected_blocks],
                    ignore_index=True,
                )
                shared_main, shared_blocks = shared_infer_subset(
                    shared,
                    joint_stats,
                    trajectories,
                )

                for anonymous_group, true_cpi, selected in selected_blocks:
                    subset_hash = trial_subset_hash(selected)
                    shared_values = {
                        **shared_main,
                        **shared_blocks[anonymous_group],
                    }
                    independent_values = infer_subset(
                        independent,
                        selected,
                        trajectories,
                    )
                    for structure, model, inferred in (
                        ("shared", shared_model, shared_values),
                        ("independent", independent_model, independent_values),
                    ):
                        predicted = float(inferred["cpi"])
                        rows.append({
                            "fit_structure": structure,
                            "model": model,
                            "uses_trajectory": shared_uses_trajectory,
                            "session_id": session_id,
                            "participant_id": participant_id,
                            "anonymous_block": anonymous_group,
                            "true_cpi": true_cpi,
                            "n_trials_per_block": n_trials,
                            "repeat": repeat,
                            "trial_subset_sha256": subset_hash,
                            "predicted_cpi": predicted,
                            "signed_error": predicted - true_cpi,
                            "absolute_error": abs(predicted - true_cpi),
                            **{
                                parameter: float(inferred[parameter])
                                for parameter in (*SHARED_PARAMETERS, *BLOCK_PARAMETERS)
                            },
                        })

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("No comparison was run; requested trial counts are too large")
    summary = summarize_predictions(predictions)
    ordering = order_diagnostics(summary)
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output_dir / "shared_vs_independent_predictions.csv", index=False)
    summary.to_csv(output_dir / "shared_vs_independent_summary.csv", index=False)
    ordering.to_csv(output_dir / "shared_vs_independent_ordering.csv", index=False)
    pd.DataFrame(subset_manifest).to_csv(
        output_dir / "identical_trial_subsets.csv",
        index=False,
    )
    plot_comparison(summary, output_dir / "shared_vs_independent_cpi.png")

    manifest = {
        "schema_version": 1,
        "created_local": datetime.now().astimezone().isoformat(),
        "source_export": str(export_dir.resolve()),
        "shared_model": shared_model,
        "shared_checkpoint": shared_checkpoint,
        "independent_model": independent_model,
        "independent_checkpoint": independent_checkpoint,
        "uses_trajectory": shared_uses_trajectory,
        "shared_parameters": list(SHARED_PARAMETERS),
        "block_specific_parameters": list(BLOCK_PARAMETERS),
        "grouping_source": "block_index remapped to anonymous consecutive labels",
        "true_cpi_used_as_model_input": False,
        "identical_trials_used_for_both_structures": True,
        "trial_counts": trial_counts or "maximum balanced count per participant",
        "bootstrap_repeats": bootstrap_repeats,
        "seed": seed,
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"summary": summary, "ordering": ordering, "manifest": manifest}


def parse_trial_counts(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    counts = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not counts or any(value <= 0 for value in counts):
        raise argparse.ArgumentTypeError("trial counts must be positive or 'all'")
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-model", required=True)
    parser.add_argument("--shared-checkpoint", type=int)
    parser.add_argument("--independent-model")
    parser.add_argument("--independent-checkpoint", type=int, default=200)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trial-counts", type=parse_trial_counts, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    export_dir = args.export_dir or latest_export(DEFAULT_HUMAN_ROOT)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = export_dir / (
            "shared_cpi_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    result = run_comparison(
        export_dir,
        output_dir,
        shared_model=args.shared_model,
        shared_checkpoint=args.shared_checkpoint,
        independent_model=args.independent_model,
        independent_checkpoint=args.independent_checkpoint,
        trial_counts=args.trial_counts,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    print("\nCPI ordering comparison:")
    print(result["ordering"].to_string(index=False))
    print(f"\nResults written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
