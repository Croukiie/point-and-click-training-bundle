"""Run CPI inference on an amortizer-ready human experiment export.

The command evaluates each participant/CPI block independently because all
trials passed to one amortizer inference must share the same latent parameters.

Example:
    python -m src.experiment.infer_human_cpi
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..inference.pnc_inferer import PnCInferer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_ROOT = ROOT / "data" / "human_experiment"
DEFAULT_MODELS = (
    "PTE_CPI_CURVATURE_ONLY_24FHD_08041142",
    "PTE_CPI_CURVATURE_24FHD_07300956",
)
TRAJECTORY_COLUMNS = (
    "timestamp",
    "timediff",
    "cursor_pos_x",
    "cursor_pos_y",
    "target_rel_pos_x",
    "target_rel_pos_y",
)

PARAMETER_METADATA = {
    "theta_m": ("Motor noise", "human_motor_control"),
    "theta_c": ("Timing noise", "human_motor_control"),
    "rew_succ": ("Success reward", "behavioral_objective"),
    "rew_fail": ("Failure penalty", "behavioral_objective"),
    "time_penalty": ("Time cost", "behavioral_objective"),
    "effort_cost": ("Effort cost", "behavioral_objective"),
    "mouse_rotation_gain_x": ("Rotation gain X", "mouse_mapping_model"),
    "mouse_rotation_gain_y": ("Rotation gain Y", "mouse_mapping_model"),
    "mouse_plan_correction": ("Plan correction", "mouse_mapping_model"),
    "cpi": ("Mouse CPI", "physical_mouse_setting"),
}

MOUSE_MODEL_PARAMETERS = (
    "cpi",
    "mouse_rotation_gain_x",
    "mouse_rotation_gain_y",
    "mouse_plan_correction",
)

HUMAN_MODEL_PARAMETERS = (
    "theta_m",
    "theta_c",
    "rew_succ",
    "rew_fail",
    "time_penalty",
    "effort_cost",
)


def latest_export(human_root: Path) -> Path:
    export_root = human_root / "amortizer_ready"
    candidates = sorted(
        (
            item
            for item in export_root.iterdir()
            if item.is_dir()
            and (item / "human_amortizer_stats.csv").is_file()
            and (item / "human_amortizer_trajectories.csv").is_file()
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ) if export_root.exists() else []
    if not candidates:
        raise FileNotFoundError(
            "No amortizer-ready human export was found. Run "
            "python -m src.experiment.export_human_amortizer_data first."
        )
    return candidates[0]


def parse_trial_counts(value: str) -> list[int]:
    counts = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("trial counts must be positive integers")
    return counts


def trajectory_lookup(path: Path) -> dict[str, pd.DataFrame]:
    data = pd.read_csv(path)
    missing = set(("trial_uid", "sample_index", *TRAJECTORY_COLUMNS)) - set(data.columns)
    if missing:
        raise ValueError(f"Trajectory export is missing columns: {sorted(missing)}")
    return {
        str(uid): group.sort_values("sample_index")[list(TRAJECTORY_COLUMNS)].reset_index(drop=True)
        for uid, group in data.groupby("trial_uid", sort=False)
    }


def model_uses_trajectory(inferer: PnCInferer) -> bool:
    return not bool(inferer.data_normalization.traj.ignore_traj)


def infer_subset(
    inferer: PnCInferer,
    stats: pd.DataFrame,
    trajectories: dict[str, pd.DataFrame] | None,
) -> dict[str, float]:
    if model_uses_trajectory(inferer):
        if trajectories is None:
            raise ValueError("This amortizer requires trajectory data")
        trial_trajectories = []
        for uid in stats["trial_uid"].astype(str):
            if uid not in trajectories:
                raise ValueError(f"Missing trajectory for {uid}")
            trial_trajectories.append(trajectories[uid])
        normalized_stats, normalized_trajectories = inferer.process_data(
            stats, trial_trajectories
        )
    else:
        normalized_stats = inferer.process_data(stats, None)
        normalized_trajectories = None
    inferred = inferer.infer(
        normalized_stats,
        normalized_trajectories,
        type="mode",
        return_z=False,
        return_w=True,
        return_dict=True,
    )
    return {name: float(value) for name, value in inferred.items()}


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    grouping = ["model", "uses_trajectory", "participant_id", "true_cpi", "n_trials"]
    return (
        predictions.groupby(grouping, as_index=False)
        .agg(
            repeats=("repeat", "count"),
            predicted_cpi_mean=("predicted_cpi", "mean"),
            predicted_cpi_median=("predicted_cpi", "median"),
            predicted_cpi_std=("predicted_cpi", "std"),
            predicted_cpi_min=("predicted_cpi", "min"),
            predicted_cpi_max=("predicted_cpi", "max"),
            signed_error_mean=("signed_error", "mean"),
            absolute_error_mean=("absolute_error", "mean"),
        )
        .sort_values(["model", "participant_id", "n_trials", "true_cpi"])
        .reset_index(drop=True)
    )


def summarize_all_parameters(
    predictions: pd.DataFrame, parameter_names: list[str]
) -> pd.DataFrame:
    grouping = ["model", "uses_trajectory", "participant_id", "true_cpi", "n_trials"]
    long_data = predictions.melt(
        id_vars=[*grouping, "repeat"],
        value_vars=parameter_names,
        var_name="parameter",
        value_name="inferred_value",
    )
    summary = (
        long_data.groupby([*grouping, "parameter"], as_index=False)
        .agg(
            repeats=("repeat", "count"),
            inferred_mean=("inferred_value", "mean"),
            inferred_median=("inferred_value", "median"),
            inferred_std=("inferred_value", "std"),
            inferred_min=("inferred_value", "min"),
            inferred_max=("inferred_value", "max"),
        )
    )
    summary["parameter_label"] = summary["parameter"].map(
        lambda name: PARAMETER_METADATA.get(name, (name, "other"))[0]
    )
    summary["parameter_group"] = summary["parameter"].map(
        lambda name: PARAMETER_METADATA.get(name, (name, "other"))[1]
    )
    return summary.sort_values(
        ["parameter_group", "parameter", "model", "participant_id", "n_trials", "true_cpi"]
    ).reset_index(drop=True)


def full_trial_parameter_table(
    predictions: pd.DataFrame, parameter_names: list[str]
) -> pd.DataFrame:
    grouping = ["model", "participant_id", "true_cpi"]
    maximum = predictions.groupby(grouping)["n_trials"].transform("max")
    columns = [
        "model",
        "uses_trajectory",
        "participant_id",
        "true_cpi",
        "n_trials",
        *parameter_names,
    ]
    return predictions.loc[predictions["n_trials"].eq(maximum), columns].sort_values(
        ["model", "participant_id", "true_cpi"]
    ).reset_index(drop=True)


def summarize_observed_mouse_data(
    stats: pd.DataFrame, human_root: Path
) -> pd.DataFrame:
    """Keep measured protocol/hardware context separate from latent inference."""
    rows: list[dict[str, Any]] = []
    raw_cache: dict[str, pd.DataFrame] = {}
    for (session_id, participant_id, cpi), group in stats.groupby(
        ["session_id", "participant_id", "cpi"], sort=True
    ):
        session_dir = human_root / str(session_id)
        metadata: dict[str, Any] = {}
        session_path = session_dir / "session.json"
        if session_path.is_file():
            metadata = json.loads(session_path.read_text(encoding="utf-8"))
        hardware = metadata.get("hardware", {})
        task_config = metadata.get("task", {}).get("config", {}).get("experiment", {})

        polling_values = pd.DataFrame()
        trials_path = session_dir / "trials.csv"
        if trials_path.is_file():
            trials = pd.read_csv(trials_path, low_memory=False)
            selected_ids = set(pd.to_numeric(group["trial_id"], errors="coerce").dropna())
            trial_ids = pd.to_numeric(trials.get("trial_id"), errors="coerce")
            trial_cpi = pd.to_numeric(trials.get("cpi"), errors="coerce")
            polling_values = trials[
                trial_ids.isin(selected_ids) & trial_cpi.eq(float(cpi))
            ].copy()

        trial_mean_interval_ns = pd.to_numeric(
            polling_values.get("mean_polling_interval_ns", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        event_rate = pd.to_numeric(
            polling_values.get("raw_event_rate_hz", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        event_count = pd.to_numeric(
            polling_values.get("raw_event_count", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        raw_path = session_dir / "raw_mouse_events.csv"
        if str(session_id) not in raw_cache:
            if raw_path.is_file():
                raw_cache[str(session_id)] = pd.read_csv(
                    raw_path,
                    usecols=["trial_id", "cpi", "dt_ns"],
                    low_memory=False,
                )
            else:
                raw_cache[str(session_id)] = pd.DataFrame()
        raw_events = raw_cache[str(session_id)]
        raw_intervals = pd.Series(dtype=float)
        if not raw_events.empty:
            raw_trial_ids = pd.to_numeric(raw_events["trial_id"], errors="coerce")
            raw_cpi = pd.to_numeric(raw_events["cpi"], errors="coerce")
            raw_intervals = pd.to_numeric(
                raw_events.loc[
                    raw_trial_ids.isin(selected_ids) & raw_cpi.eq(float(cpi)), "dt_ns"
                ],
                errors="coerce",
            )
            raw_intervals = raw_intervals[raw_intervals > 0].dropna()
        median_raw_interval_ns = (
            float(raw_intervals.median()) if not raw_intervals.empty else np.nan
        )
        median_trial_mean_interval_ns = (
            float(trial_mean_interval_ns.median())
            if not trial_mean_interval_ns.empty
            else np.nan
        )

        def mean_column(name: str) -> float:
            values = pd.to_numeric(group.get(name), errors="coerce").dropna()
            return float(values.mean()) if not values.empty else np.nan

        rows.append({
            "session_id": session_id,
            "participant_id": participant_id,
            "cpi_condition": float(cpi),
            "n_trials": len(group),
            "mouse_model": hardware.get("mouse_model", ""),
            "cpi_setting_method": hardware.get("cpi_setting_method", ""),
            "declared_mouse_polling_hz": hardware.get("declared_mouse_polling_hz", np.nan),
            "observed_wm_input_delivery_hz": (
                1e9 / median_raw_interval_ns if median_raw_interval_ns > 0 else np.nan
            ),
            "median_wm_input_interval_ms": median_raw_interval_ns / 1e6,
            "median_trial_mean_polling_interval_ms": median_trial_mean_interval_ns / 1e6,
            "median_raw_event_rate_hz": (
                float(event_rate.median()) if not event_rate.empty else np.nan
            ),
            "median_raw_event_count": (
                float(event_count.median()) if not event_count.empty else np.nan
            ),
            "windows_pointer_speed": hardware.get("windows_pointer_speed", ""),
            "pointer_acceleration": hardware.get("pointer_acceleration", ""),
            "input_backend": metadata.get("input_backend", hardware.get("input_backend", "")),
            "protocol_mouse_gain": task_config.get("mouse_gain", np.nan),
            "protocol_mouse_reference_cpi": task_config.get("mouse_reference_cpi", np.nan),
            "movement_time_mean_s": mean_column("movement_time"),
            "target_hit_rate": mean_column("target_hit"),
            "click_distance_mean_m": mean_column("click_distance"),
            "cursor_path_length_mean_m": mean_column("traverse_length"),
            "max_cursor_speed_mean_m_s": mean_column("max_cursor_speed"),
            "total_abs_turn_mean_rad": mean_column("cursor_total_abs_turn_rad"),
            "mean_abs_turn_mean_rad": mean_column("cursor_mean_abs_turn_rad"),
            "max_deviation_normalized_mean": mean_column("cursor_max_deviation_normalized"),
            "path_excess_ratio_mean": mean_column("cursor_path_excess_ratio"),
        })
    return pd.DataFrame(rows)


def order_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = ["model", "uses_trajectory", "participant_id", "n_trials"]
    for keys, group in summary.groupby(grouping, sort=False):
        ordered = group.sort_values("true_cpi")
        true_values = ordered["true_cpi"].to_numpy(dtype=float)
        predicted = ordered["predicted_cpi_mean"].to_numpy(dtype=float)
        rows.append({
            **dict(zip(grouping, keys)),
            "true_cpi_order": ";".join(f"{value:g}" for value in true_values),
            "predicted_cpi_order": ";".join(f"{value:.2f}" for value in predicted),
            "strictly_increasing": bool(np.all(np.diff(predicted) > 0)),
            "pearson_order_correlation": (
                float(np.corrcoef(true_values, predicted)[0, 1])
                if len(true_values) >= 2 and np.std(predicted) > 0
                else np.nan
            ),
        })
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    models = list(summary["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, summary["n_trials"].nunique()))
    for axis, model in zip(axes[0], models):
        subset = summary[summary["model"] == model]
        for color, (n_trials, group) in zip(colors, subset.groupby("n_trials")):
            group = group.sort_values("true_cpi")
            error = group["predicted_cpi_std"].fillna(0.0)
            axis.errorbar(
                group["true_cpi"],
                group["predicted_cpi_mean"],
                yerr=error,
                marker="o",
                linewidth=2,
                capsize=3,
                color=color,
                label=f"n={n_trials}",
            )
        lower = min(350.0, float(subset["true_cpi"].min()))
        upper = max(1650.0, float(subset["true_cpi"].max()))
        axis.plot([lower, upper], [lower, upper], "--", color="0.45", label="ideal")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel("True CPI")
        axis.set_ylabel("Inferred CPI")
        axis.set_title(model)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Human CPI recovery")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_group(
    summary: pd.DataFrame,
    parameter_names: tuple[str, ...],
    path: Path,
    title: str,
) -> None:
    available = [name for name in parameter_names if name in set(summary["parameter"])]
    if not available:
        return
    max_trials = int(summary["n_trials"].max())
    plot_data = summary[summary["n_trials"] == max_trials]
    ncols = 2 if len(available) <= 4 else 3
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.3 * nrows), squeeze=False)
    for axis, parameter in zip(axes.flat, available):
        subset = plot_data[plot_data["parameter"] == parameter]
        for (model, participant_id), group in subset.groupby(
            ["model", "participant_id"], sort=False
        ):
            group = group.sort_values("true_cpi")
            axis.errorbar(
                group["true_cpi"],
                group["inferred_mean"],
                yerr=group["inferred_std"].fillna(0.0),
                marker="o",
                linewidth=2,
                capsize=3,
                label=f"{model} / {participant_id}",
            )
        if parameter == "cpi":
            lower = float(subset["true_cpi"].min())
            upper = float(subset["true_cpi"].max())
            axis.plot([lower, upper], [lower, upper], "--", color="0.45", label="ideal CPI")
        axis.set_title(PARAMETER_METADATA.get(parameter, (parameter, ""))[0])
        axis.set_xlabel("Physical CPI condition")
        axis.set_ylabel("Inferred value")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    for axis in axes.flat[len(available):]:
        axis.set_visible(False)
    fig.suptitle(f"{title} (n={max_trials} trials per CPI block)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_inference(
    export_dir: Path,
    output_dir: Path,
    *,
    models: list[str],
    checkpoint: int,
    trial_counts: list[int],
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    if bootstrap_repeats <= 0:
        raise ValueError("bootstrap_repeats must be positive")
    stats = pd.read_csv(export_dir / "human_amortizer_stats.csv")
    required = {"trial_uid", "participant_id", "cpi"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"Statistics export is missing columns: {sorted(missing)}")
    trajectories: dict[str, pd.DataFrame] | None = None
    rng = np.random.default_rng(seed)
    prediction_rows: list[dict[str, Any]] = []
    common_parameter_names: list[str] | None = None

    for model in models:
        inferer = PnCInferer(model, ckpt=checkpoint, verbose=True)
        uses_trajectory = model_uses_trajectory(inferer)
        if uses_trajectory and trajectories is None:
            trajectories = trajectory_lookup(
                export_dir / "human_amortizer_trajectories.csv"
            )
        parameter_names = list(inferer.simulator.total_target_param)
        if common_parameter_names is None:
            common_parameter_names = parameter_names
        elif parameter_names != common_parameter_names:
            raise ValueError("Selected amortizers do not infer the same parameter set")
        for (participant_id, true_cpi), group in stats.groupby(
            ["participant_id", "cpi"], sort=True
        ):
            group = group.reset_index(drop=True)
            for n_trials in trial_counts:
                if n_trials > len(group):
                    continue
                repeats = 1 if n_trials == len(group) else bootstrap_repeats
                for repeat in range(repeats):
                    if n_trials == len(group):
                        subset = group
                    else:
                        indices = np.sort(
                            rng.choice(len(group), size=n_trials, replace=False)
                        )
                        subset = group.iloc[indices].reset_index(drop=True)
                    inferred = infer_subset(inferer, subset, trajectories)
                    predicted_cpi = inferred["cpi"]
                    prediction_rows.append({
                        "model": model,
                        "uses_trajectory": uses_trajectory,
                        "participant_id": participant_id,
                        "true_cpi": float(true_cpi),
                        "n_trials": n_trials,
                        "repeat": repeat,
                        "predicted_cpi": predicted_cpi,
                        "signed_error": predicted_cpi - float(true_cpi),
                        "absolute_error": abs(predicted_cpi - float(true_cpi)),
                        **{name: inferred[name] for name in parameter_names},
                    })

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ValueError("No inference was run; requested trial counts exceed block sizes")
    if common_parameter_names is None:
        raise ValueError("No amortizer parameters were found")
    summary = summarize_predictions(predictions)
    parameter_summary = summarize_all_parameters(predictions, common_parameter_names)
    full_parameters = full_trial_parameter_table(predictions, common_parameter_names)
    observed_mouse = summarize_observed_mouse_data(stats, DEFAULT_HUMAN_ROOT)
    order = order_diagnostics(summary)
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output_dir / "human_cpi_predictions.csv", index=False)
    summary.to_csv(output_dir / "human_cpi_summary.csv", index=False)
    parameter_summary.to_csv(output_dir / "human_all_parameters_summary.csv", index=False)
    full_parameters.to_csv(output_dir / "human_inferred_parameters_full_trials.csv", index=False)
    observed_mouse.to_csv(output_dir / "human_observed_mouse_metrics.csv", index=False)
    order.to_csv(output_dir / "human_cpi_order_diagnostics.csv", index=False)
    plot_summary(summary, output_dir / "human_cpi_inference.png")
    plot_parameter_group(
        parameter_summary,
        MOUSE_MODEL_PARAMETERS,
        output_dir / "human_mouse_model_parameters.png",
        "Mouse-related amortizer parameters",
    )
    plot_parameter_group(
        parameter_summary,
        HUMAN_MODEL_PARAMETERS,
        output_dir / "human_behavioral_parameters.png",
        "Human motor and behavioral parameters",
    )

    manifest = {
        "schema_version": 1,
        "created_local": datetime.now().astimezone().isoformat(),
        "source_export": str(export_dir.resolve()),
        "models": models,
        "checkpoint": checkpoint,
        "trial_counts": trial_counts,
        "bootstrap_repeats_for_subsets": bootstrap_repeats,
        "seed": seed,
        "prediction_rows": len(predictions),
        "files": {
            "predictions": "human_cpi_predictions.csv",
            "summary": "human_cpi_summary.csv",
            "all_parameter_summary": "human_all_parameters_summary.csv",
            "full_trial_parameters": "human_inferred_parameters_full_trials.csv",
            "observed_mouse_metrics": "human_observed_mouse_metrics.csv",
            "order_diagnostics": "human_cpi_order_diagnostics.csv",
            "cpi_plot": "human_cpi_inference.png",
            "mouse_parameter_plot": "human_mouse_model_parameters.png",
            "behavioral_parameter_plot": "human_behavioral_parameters.png",
        },
        "interpretation_note": (
            "Only CPI is a physical sensor setting. Rotation gains, plan correction, "
            "motor noise, timing noise, and reward/cost terms are simulator latent "
            "parameters. Hardware/protocol fields and movement metrics are observed, "
            "not inferred."
        ),
    }
    (output_dir / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "summary": summary,
        "parameter_summary": parameter_summary,
        "full_parameters": full_parameters,
        "observed_mouse": observed_mouse,
        "order": order,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--checkpoint", type=int, default=200)
    parser.add_argument("--trial-counts", default="16,32,64,128")
    parser.add_argument("--bootstrap-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    export_dir = args.export_dir or latest_export(DEFAULT_HUMAN_ROOT)
    trial_counts = parse_trial_counts(args.trial_counts)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = export_dir / f"inference_{timestamp}"
    result = run_inference(
        export_dir,
        output_dir,
        models=args.models,
        checkpoint=args.checkpoint,
        trial_counts=trial_counts,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    summary: pd.DataFrame = result["summary"]
    full_parameters: pd.DataFrame = result["full_parameters"]
    observed_mouse: pd.DataFrame = result["observed_mouse"]
    order: pd.DataFrame = result["order"]
    max_trials = int(summary["n_trials"].max())
    print("\nCPI inference at the largest available trial count:")
    print(
        summary[summary["n_trials"] == max_trials][
            ["model", "participant_id", "true_cpi", "predicted_cpi_mean", "absolute_error_mean"]
        ].to_string(index=False)
    )
    failing = order[
        (order["n_trials"] == max_trials) & (~order["strictly_increasing"])
    ]
    if not failing.empty:
        print("\nWARNING: inferred CPI order is not strictly increasing:")
        print(
            failing[
                ["model", "participant_id", "true_cpi_order", "predicted_cpi_order"]
            ].to_string(index=False)
        )
    print("\nAll inferred parameters at the largest available trial count:")
    print(full_parameters.to_string(index=False))
    print("\nObserved mouse/protocol context (not inferred):")
    print(
        observed_mouse[
            [
                "participant_id",
                "cpi_condition",
                "mouse_model",
                "declared_mouse_polling_hz",
                "observed_wm_input_delivery_hz",
                "windows_pointer_speed",
                "pointer_acceleration",
                "protocol_mouse_gain",
            ]
        ].to_string(index=False)
    )
    print(f"\nResults written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
