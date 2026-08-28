"""Audit all inferred parameters and posterior-predictive human behavior.

This diagnostic consumes an existing shared-CPI human comparison directory.
It does not retrain the RL policy or amortizer.  Simulated recovery is read
from the saved validation pickles, human stability is calculated across trial
subsets, and posterior-predictive trials are replayed with fitted parameters.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import yaml

from ..agent.pnc_simulator import PnCSimulator
from ..utils.mymath import recovery_metrics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON = (
    ROOT
    / "data/human_experiment/amortizer_ready/"
    "v3_elisa_rawcount125_20260824_unconfirmed_labels/"
    "rawcount_vs_speedpath_comparison_20260824"
)
PPC_FEATURES = (
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
    "raw_count_moving_bins",
    "raw_count_mean_per_moving_bin",
    "raw_count_p90_per_moving_bin",
    "raw_count_max_per_bin",
    "raw_count_path_per_second",
)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_map(predictions: pd.DataFrame) -> dict[str, str]:
    pairs = predictions[["model_variant", "model"]].drop_duplicates()
    duplicated = pairs.groupby("model_variant")["model"].nunique()
    if (duplicated > 1).any():
        raise ValueError("Each model variant must map to exactly one model directory")
    return dict(zip(pairs["model_variant"], pairs["model"]))


def _parameter_contract(model: str) -> tuple[list[str], list[str], dict]:
    root = ROOT / "data/amortizer_cond/models" / model
    design = _load_yaml(root / "shared_fit_design.yaml")
    simulator = _load_yaml(root / "simulator_setting.yaml")
    return (
        list(design["shared_parameters"]),
        list(design["block_specific_parameters"]),
        simulator,
    )


def _parameter_bounds(rl_model: str) -> dict[str, tuple[float, float]]:
    env = _load_yaml(ROOT / "data/pnc_model" / rl_model / "env_info.yaml")
    return {
        name: (float(setting["min"]), float(setting["max"]))
        for name, setting in env["user_config"]["param_modul"].items()
        if isinstance(setting, dict) and "min" in setting and "max" in setting
    }


def _latest_recovery_pickle(model: str, trial_count: int) -> Path:
    result_root = ROOT / "data/amortizer_cond/models" / model / "results"
    candidates = []
    for directory in result_root.glob("iter*"):
        try:
            iteration = int(directory.name.removeprefix("iter"))
        except ValueError:
            continue
        source = directory / f"trial{trial_count:03d}.pkl"
        if source.exists():
            candidates.append((iteration, source))
    if not candidates:
        raise FileNotFoundError(
            f"No trial{trial_count:03d}.pkl recovery file exists for {model}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def simulated_recovery(
    models: dict[str, str], trial_count: int, bounds: dict[str, tuple[float, float]], output: Path
) -> pd.DataFrame:
    rows = []
    for label, model in models.items():
        shared, block, _ = _parameter_contract(model)
        source = _latest_recovery_pickle(model, trial_count)
        with source.open("rb") as handle:
            true_main, true_block, predicted_main, predicted_block = pickle.load(handle)
        arrays = [
            (name, true_main[:, index], predicted_main[:, index])
            for index, name in enumerate(shared)
        ] + [
            (name, true_block[:, :, index].reshape(-1), predicted_block[:, index])
            if true_block.ndim == 3
            else (name, true_block[:, index].reshape(-1), predicted_block[:, index])
            for index, name in enumerate(block)
        ]

        fig, axes = plt.subplots(2, 5, figsize=(17, 7), squeeze=False)
        for axis, (name, true, predicted) in zip(axes.ravel(), arrays):
            metric = recovery_metrics(true, predicted)
            residual = np.asarray(predicted) - np.asarray(true)
            low, high = bounds.get(name, (float(np.min(true)), float(np.max(true))))
            width = high - low
            rows.append({
                "model_variant": label,
                "model": model,
                "trial_count": trial_count,
                "parameter": name,
                "n": len(np.asarray(true).reshape(-1)),
                "predictive_r2": metric["r2"],
                "pearson_r": metric["pearson_r"],
                "slope": metric["slope"],
                "intercept": metric["intercept"],
                "mae": metric["mae"],
                "rmse": metric["rmse"],
                "bias": float(np.mean(residual)),
                "normalized_rmse": metric["rmse"] / width if width else np.nan,
                "range_min": low,
                "range_max": high,
                "recovery_pickle": str(source.resolve()),
            })
            axis.scatter(true, predicted, s=13, alpha=0.55)
            axis.plot([low, high], [low, high], "--", color="black", linewidth=1)
            axis.set_title(f"{name}\nR2={metric['r2']:.2f}, r={metric['pearson_r']:.2f}")
            axis.set_xlabel("True")
            axis.set_ylabel("Inferred")
            axis.grid(alpha=0.2)
        fig.suptitle(f"Simulated recovery: {label} ({trial_count} trials/block)")
        fig.tight_layout()
        fig.savefig(output / f"simulated_parameter_recovery_{label}.png", dpi=180)
        plt.close(fig)
    return pd.DataFrame(rows)


def human_stability(
    predictions: pd.DataFrame,
    shared_parameters: list[str],
    bounds: dict[str, tuple[float, float]],
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identity = [
        "model_variant", "model", "session_id", "participant_id",
        "n_trials_per_block", "repeat",
    ]
    spread_rows = []
    unique_rows = []
    for keys, group in predictions.groupby(identity, sort=True):
        row = dict(zip(identity, keys))
        for parameter in shared_parameters:
            values = group[parameter].to_numpy(dtype=float)
            row[parameter] = float(values[0])
            spread_rows.append({
                **dict(zip(identity, keys)),
                "parameter": parameter,
                "within_joint_fit_block_spread": float(np.ptp(values)),
            })
        unique_rows.append(row)
    unique = pd.DataFrame(unique_rows)

    long = unique.melt(
        id_vars=identity,
        value_vars=shared_parameters,
        var_name="parameter",
        value_name="estimate",
    )
    stability = (
        long.groupby(
            ["model_variant", "model", "session_id", "participant_id", "n_trials_per_block", "parameter"],
            as_index=False,
        )
        .agg(
            repeats=("repeat", "count"),
            estimate_mean=("estimate", "mean"),
            estimate_sd=("estimate", "std"),
            estimate_q025=("estimate", lambda x: x.quantile(0.025)),
            estimate_q975=("estimate", lambda x: x.quantile(0.975)),
        )
    )
    stability["range_min"] = stability["parameter"].map(lambda x: bounds[x][0])
    stability["range_max"] = stability["parameter"].map(lambda x: bounds[x][1])
    width = stability["range_max"] - stability["range_min"]
    stability["normalized_sd"] = stability["estimate_sd"] / width
    position = (stability["estimate_mean"] - stability["range_min"]) / width
    stability["normalized_position"] = position
    stability["within_5pct_of_boundary"] = (position <= 0.05) | (position >= 0.95)

    max_count = int(stability["n_trials_per_block"].max())
    reference = stability[
        stability["n_trials_per_block"] == max_count
    ][
        ["model_variant", "session_id", "participant_id", "parameter", "estimate_mean"]
    ].rename(columns={"estimate_mean": "max_trial_count_estimate"})
    stability = stability.merge(
        reference,
        on=["model_variant", "session_id", "participant_id", "parameter"],
        how="left",
    )
    stability["absolute_change_to_max_trial_count"] = (
        stability["estimate_mean"] - stability["max_trial_count_estimate"]
    ).abs()
    stability["normalized_change_to_max_trial_count"] = (
        stability["absolute_change_to_max_trial_count"]
        / (stability["range_max"] - stability["range_min"])
    )
    at_max = stability[stability["n_trials_per_block"] == max_count]
    agreement_rows = []
    for keys, group in at_max.groupby(["session_id", "participant_id", "parameter"]):
        if len(group) < 2:
            continue
        first, second = group.sort_values("model_variant").iloc[:2].to_dict("records")
        parameter = keys[2]
        parameter_width = bounds[parameter][1] - bounds[parameter][0]
        agreement_rows.append({
            "session_id": keys[0],
            "participant_id": keys[1],
            "parameter": parameter,
            "model_a": first["model_variant"],
            "estimate_a": first["estimate_mean"],
            "model_b": second["model_variant"],
            "estimate_b": second["estimate_mean"],
            "absolute_difference": abs(first["estimate_mean"] - second["estimate_mean"]),
            "normalized_absolute_difference": abs(first["estimate_mean"] - second["estimate_mean"]) / parameter_width,
        })
    agreement = pd.DataFrame(agreement_rows)

    for (session, participant), group in stability.groupby(["session_id", "participant_id"]):
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), squeeze=False)
        for axis, parameter in zip(axes.ravel(), shared_parameters):
            subset = group[group["parameter"] == parameter]
            for label, line in subset.groupby("model_variant"):
                line = line.sort_values("n_trials_per_block")
                axis.errorbar(
                    line["n_trials_per_block"], line["estimate_mean"],
                    yerr=line["estimate_sd"].fillna(0), marker="o", capsize=3,
                    label=label,
                )
            axis.axhline(bounds[parameter][0], color="0.7", linewidth=0.8)
            axis.axhline(bounds[parameter][1], color="0.7", linewidth=0.8)
            axis.set_title(parameter)
            axis.set_xlabel("Trials per CPI block")
            axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Human shared-parameter stability: {participant} / {session}")
        fig.tight_layout()
        safe = "".join(character if character.isalnum() else "_" for character in f"{participant}_{session}")
        fig.savefig(output / f"human_parameter_stability_{safe}.png", dpi=180)
        plt.close(fig)
    return stability, pd.DataFrame(spread_rows), agreement


def _centered_task(row: pd.Series) -> dict:
    delta = np.array(
        [float(row["target_rel_pos_x0"]), float(row["target_rel_pos_y0"])],
        dtype=float,
    )
    return {
        "cursor_pos": (-delta / 2).tolist(),
        "target_pos": (delta / 2).tolist(),
        "target_rad": float(row["target_radius"]),
        "target_spd": 0.0,
        "target_dir": 0.0,
    }


def posterior_predictive(
    predictions: pd.DataFrame,
    human_stats: pd.DataFrame,
    models: dict[str, str],
    shared_parameters: list[str],
    trials_per_block: int,
    seed: int,
    device: str,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    max_count = int(predictions["n_trials_per_block"].max())
    fitted = predictions[predictions["n_trials_per_block"] == max_count]
    human_rows = []
    simulation_rows = []
    matched_trials = {}
    for keys, block in human_stats.groupby(
        ["session_id", "participant_id", "cpi"], sort=True
    ):
        block = block.reset_index(drop=True)
        count = min(trials_per_block, len(block))
        indices = np.sort(rng.choice(len(block), size=count, replace=False))
        matched_trials[keys] = block.iloc[indices].copy().reset_index(drop=True)

    for label, model in models.items():
        _, _, simulator_config = _parameter_contract(model)
        rl_model = str(simulator_config["model_name"])
        rl_checkpoint = simulator_config.get("ckpt", "best")
        simulator = PnCSimulator(
            rl_model, ckpt=rl_checkpoint, policy_device=device
        )
        for (session, participant), participant_stats in human_stats.groupby(
            ["session_id", "participant_id"], sort=True
        ):
            fit = fitted[
                (fitted["model_variant"] == label)
                & (fitted["session_id"] == session)
                & (fitted["participant_id"] == participant)
            ]
            if fit.empty:
                continue
            shared = {
                parameter: float(fit[parameter].mean())
                for parameter in shared_parameters
            }
            selected_parts = []
            task_list = []
            user_param_list = []
            labels = []
            for true_cpi, block in participant_stats.groupby("cpi", sort=True):
                selected = matched_trials[(session, participant, true_cpi)].copy()
                predicted_cpi = float(
                    fit[np.isclose(fit["true_cpi"], float(true_cpi))]["predicted_cpi"].mean()
                )
                for _, trial in selected.iterrows():
                    params = {**shared, "cpi": predicted_cpi}
                    user_param_list.append({"param_w": params})
                    task_list.append(_centered_task(trial))
                    labels.append(float(true_cpi))
                selected["source"] = "human"
                selected["model_variant"] = label
                selected["true_cpi"] = float(true_cpi)
                selected_parts.append(selected)

            simulator.simulate(
                user_param_list=user_param_list,
                task_list=task_list,
                overwrite_existing_simul=True,
                verbose=False,
            )
            simulated = simulator.get_simulation_result(downsample=20).reset_index(drop=True)
            simulated["source"] = "posterior_predictive_simulation"
            simulated["model_variant"] = label
            simulated["session_id"] = session
            simulated["participant_id"] = participant
            simulated["true_cpi"] = labels
            human = pd.concat(selected_parts, ignore_index=True)
            human["session_id"] = session
            human["participant_id"] = participant
            human_rows.append(human)
            simulation_rows.append(simulated)

    human_all = pd.concat(human_rows, ignore_index=True)
    simulation_all = pd.concat(simulation_rows, ignore_index=True)
    common = [feature for feature in PPC_FEATURES if feature in human_all and feature in simulation_all]
    trial_level = pd.concat(
        [
            human_all[["source", "model_variant", "session_id", "participant_id", "true_cpi", *common]],
            simulation_all[["source", "model_variant", "session_id", "participant_id", "true_cpi", *common]],
        ],
        ignore_index=True,
    )

    rows = []
    slopes = []
    keys = ["model_variant", "session_id", "participant_id"]
    for group_keys, group in trial_level.groupby(keys, sort=True):
        label, session, participant = group_keys
        for feature in common:
            cpi_means = {}
            for true_cpi, cpi_group in group.groupby("true_cpi", sort=True):
                human_values = cpi_group[cpi_group["source"] == "human"][feature].dropna().to_numpy(float)
                sim_values = cpi_group[cpi_group["source"] != "human"][feature].dropna().to_numpy(float)
                pooled = math.sqrt((np.var(human_values) + np.var(sim_values)) / 2) if len(human_values) and len(sim_values) else np.nan
                rows.append({
                    "model_variant": label,
                    "session_id": session,
                    "participant_id": participant,
                    "true_cpi": true_cpi,
                    "feature": feature,
                    "human_n": len(human_values),
                    "human_mean": float(np.mean(human_values)),
                    "human_sd": float(np.std(human_values)),
                    "simulation_n": len(sim_values),
                    "simulation_mean": float(np.mean(sim_values)),
                    "simulation_sd": float(np.std(sim_values)),
                    "standardized_mean_difference": (
                        float((np.mean(sim_values) - np.mean(human_values)) / pooled)
                        if pooled and np.isfinite(pooled) else np.nan
                    ),
                    "ks_statistic": float(ks_2samp(human_values, sim_values).statistic),
                    "ks_p_value": float(ks_2samp(human_values, sim_values).pvalue),
                })
                cpi_means[true_cpi] = (float(np.mean(human_values)), float(np.mean(sim_values)))
            ordered = sorted(cpi_means)
            human_slope = float(np.polyfit(ordered, [cpi_means[c][0] for c in ordered], 1)[0])
            simulation_slope = float(np.polyfit(ordered, [cpi_means[c][1] for c in ordered], 1)[0])
            slopes.append({
                "model_variant": label,
                "session_id": session,
                "participant_id": participant,
                "feature": feature,
                "human_cpi_slope": human_slope,
                "simulation_cpi_slope": simulation_slope,
                "slope_direction_agrees": bool(np.sign(human_slope) == np.sign(simulation_slope)),
            })

    summary = pd.DataFrame(rows)
    slope_frame = pd.DataFrame(slopes)
    for label, model_data in summary.groupby("model_variant"):
        features = list(dict.fromkeys(model_data["feature"]))
        columns = 4
        rows_count = math.ceil(len(features) / columns)
        fig, axes = plt.subplots(rows_count, columns, figsize=(18, 3.8 * rows_count), squeeze=False)
        for axis, feature in zip(axes.ravel(), features):
            line = model_data[model_data["feature"] == feature].sort_values("true_cpi")
            axis.errorbar(line["true_cpi"], line["human_mean"], yerr=line["human_sd"], marker="o", capsize=2, label="human")
            axis.errorbar(line["true_cpi"], line["simulation_mean"], yerr=line["simulation_sd"], marker="s", capsize=2, label="posterior predictive")
            axis.set_title(feature)
            axis.set_xlabel("Requested CPI block")
            axis.grid(alpha=0.2)
        for axis in axes.ravel()[len(features):]:
            axis.axis("off")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Human versus fitted-model posterior predictive trends: {label}")
        fig.tight_layout()
        fig.savefig(output / f"posterior_predictive_features_{label}.png", dpi=180)
        plt.close(fig)
    return trial_level, summary, slope_frame


def write_report(
    output: Path,
    recovery: pd.DataFrame,
    stability: pd.DataFrame,
    agreement: pd.DataFrame,
    ppc: pd.DataFrame | None,
    slopes: pd.DataFrame | None,
) -> None:
    worst_recovery = recovery.sort_values("predictive_r2").head(5)
    max_count = stability["n_trials_per_block"].max()
    boundary = stability[
        (stability["n_trials_per_block"] == max_count)
        & stability["within_5pct_of_boundary"]
    ]
    lines = [
        "# Full shared-parameter and human posterior-predictive audit",
        "",
        f"Generated: {datetime.now().astimezone().isoformat()}",
        "",
        "## What is directly validated",
        "",
        "All ten parameters have known ground truth in simulation, so calibration-aware recovery metrics are reported for each one. Human ground truth exists only for CPI; the nine shared human parameters are therefore evaluated by subset stability, boundary behavior, model agreement, and posterior-predictive fit.",
        "",
        "## Weakest simulated parameter recoveries",
        "",
        "| Model | Parameter | Predictive R2 | Pearson r | Slope | Normalized RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in worst_recovery.itertuples():
        lines.append(f"| {row.model_variant} | {row.parameter} | {row.predictive_r2:.3f} | {row.pearson_r:.3f} | {row.slope:.3f} | {row.normalized_rmse:.3f} |")
    lines.extend(["", "## Human shared-parameter stability", ""])
    if boundary.empty:
        lines.append("No shared human estimate is within 5% of a prior boundary at the largest trial count.")
    else:
        names = ", ".join(sorted(set(boundary["parameter"])))
        lines.append(f"Parameters with at least one estimate within 5% of a prior boundary: {names}.")
    if not agreement.empty:
        largest = agreement.sort_values("normalized_absolute_difference", ascending=False).head(5)
        lines.extend([
            "",
            "Largest disagreements between fitted model variants:",
            "",
            "| Parameter | Model A | Model B | Difference / prior width |",
            "|---|---|---|---:|",
        ])
        for row in largest.itertuples():
            lines.append(f"| {row.parameter} | {row.model_a} | {row.model_b} | {row.normalized_absolute_difference:.3f} |")
    drift = stability[
        stability["n_trials_per_block"] < max_count
    ].sort_values("normalized_change_to_max_trial_count", ascending=False).head(5)
    if not drift.empty:
        lines.extend([
            "",
            "Largest changes relative to the largest trial subset:",
            "",
            "| Model | Trials/block | Parameter | Change / prior width |",
            "|---|---:|---|---:|",
        ])
        for row in drift.itertuples():
            lines.append(f"| {row.model_variant} | {row.n_trials_per_block} | {row.parameter} | {row.normalized_change_to_max_trial_count:.3f} |")
    if ppc is not None and slopes is not None:
        largest = ppc.assign(abs_smd=ppc["standardized_mean_difference"].abs()).sort_values("abs_smd", ascending=False).head(8)
        disagreements = slopes[~slopes["slope_direction_agrees"]]
        lines.extend([
            "",
            "## Posterior-predictive human fit",
            "",
            "Largest standardized simulation-to-human differences:",
            "",
            "| Model | CPI | Feature | Standardized difference | KS statistic |",
            "|---|---:|---|---:|---:|",
        ])
        for row in largest.itertuples():
            lines.append(f"| {row.model_variant} | {row.true_cpi:.0f} | {row.feature} | {row.standardized_mean_difference:.2f} | {row.ks_statistic:.2f} |")
        lines.extend(["", f"Opposite simulated versus human CPI slopes: {len(disagreements)} of {len(slopes)} model-feature comparisons."])
        if len(disagreements):
            lines.append("Affected features: " + ", ".join(sorted(set(disagreements["feature"]))) + ".")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Good simulated recovery is necessary but does not validate a human latent parameter. A human parameter should also be stable across trial subsets, avoid unexplained prior-boundary collapse, agree across defensible feature ablations, and reproduce held-out behavioral summaries. CPI is the only parameter for which absolute human error can currently be reported.",
        "",
        "This v3 export contains one participant session, so between-session and leave-one-participant-out stability cannot yet be estimated. The CPI labels in this historical session were not explicitly confirmed in the trial CSV; the corrected experiment recorder should be used for future confirmatory data.",
    ])
    (output / "FULL_HUMAN_INFERENCE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--simulated-trial-count", type=int, default=64)
    parser.add_argument("--ppc-trials-per-block", type=int, default=16)
    parser.add_argument("--ppc-seed", type=int, default=20260824)
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--skip-ppc", action="store_true")
    args = parser.parse_args(argv)

    comparison = args.comparison_dir.resolve()
    predictions = pd.read_csv(comparison / "human_predictions.csv")
    models = _model_map(predictions)
    first_model = next(iter(models.values()))
    shared_parameters, _, simulator_config = _parameter_contract(first_model)
    rl_model = str(simulator_config["model_name"])
    bounds = _parameter_bounds(rl_model)
    manifest_path = comparison / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    human_export = Path(manifest["human_export"])
    human_stats = pd.read_csv(human_export / "human_amortizer_stats.csv")

    output = args.output_dir or comparison / f"full_inference_audit_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    recovery = simulated_recovery(models, args.simulated_trial_count, bounds, output)
    stability, within_fit, agreement = human_stability(predictions, shared_parameters, bounds, output)
    recovery.to_csv(output / "simulated_all_parameter_recovery.csv", index=False)
    stability.to_csv(output / "human_shared_parameter_stability.csv", index=False)
    within_fit.to_csv(output / "human_shared_within_fit_block_spread.csv", index=False)
    agreement.to_csv(output / "human_shared_model_agreement.csv", index=False)

    ppc_summary = None
    slope_frame = None
    if not args.skip_ppc:
        trial_level, ppc_summary, slope_frame = posterior_predictive(
            predictions, human_stats, models, shared_parameters,
            args.ppc_trials_per_block, args.ppc_seed, args.policy_device, output,
        )
        trial_level.to_csv(output / "posterior_predictive_trial_level.csv", index=False)
        ppc_summary.to_csv(output / "posterior_predictive_feature_summary.csv", index=False)
        slope_frame.to_csv(output / "posterior_predictive_cpi_slopes.csv", index=False)
    write_report(output, recovery, stability, agreement, ppc_summary, slope_frame)
    audit_manifest = {
        "created_local": datetime.now().astimezone().isoformat(),
        "comparison_dir": str(comparison),
        "human_export": str(human_export.resolve()),
        "models": models,
        "rl_model": rl_model,
        "simulated_trial_count": args.simulated_trial_count,
        "posterior_predictive_trials_per_block": 0 if args.skip_ppc else args.ppc_trials_per_block,
        "posterior_predictive_seed": args.ppc_seed,
        "parameter_ground_truth": {
            "simulation": "available for all ten parameters",
            "human": "available only for physically measured CPI",
        },
    }
    (output / "audit_manifest.json").write_text(json.dumps(audit_manifest, indent=2), encoding="utf-8")
    print(f"OUTPUT_DIR={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
