"""Simulate the RL model on the exact target geometry of human trials."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from ..config.constant import FIELD
from ..inference.pnc_inferer import PnCInferer
from .plot_human_participant_comparison import METRICS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_ROOT = ROOT / "data" / "human_experiment"
DEFAULT_INFERENCE_ROOT = DEFAULT_HUMAN_ROOT / "amortizer_ready"
DEFAULT_AMORTIZER = "PTE_CPI_CURVATURE_24FHD_07300956"


def latest_inference(root: Path) -> Path:
    candidates = sorted(
        (
            path.parent
            for path in root.rglob("human_inferred_parameters_full_trials.csv")
            if (path.parent / "inference_manifest.json").is_file()
        ),
        key=lambda path: (path / "human_inferred_parameters_full_trials.csv").stat().st_mtime,
        reverse=True,
    ) if root.exists() else []
    if not candidates:
        raise FileNotFoundError(
            "No complete human inference was found. Run python -m "
            "src.experiment.infer_human_mouse first."
        )
    return candidates[0]


def source_export(inference_dir: Path) -> Path:
    manifest = json.loads(
        (inference_dir / "inference_manifest.json").read_text(encoding="utf-8")
    )
    return Path(manifest["source_export"])


def load_source_trials(stats: pd.DataFrame, human_root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for session_id in stats["session_id"].astype(str).unique():
        path = human_root / session_id / "trials.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source trial file: {path}")
        trials = pd.read_csv(path, low_memory=False)
        trials["session_id"] = trials["session_id"].astype(str)
        parts.append(trials)
    source = pd.concat(parts, ignore_index=True)
    source["trial_id"] = pd.to_numeric(source["trial_id"], errors="raise").astype(int)
    return source


def select_matched_trials(
    stats: pd.DataFrame,
    source: pd.DataFrame,
    trials_per_block: int,
    seed: int,
) -> pd.DataFrame:
    required_source = {
        "session_id",
        "trial_id",
        "cursor_start_x_m",
        "cursor_start_y_m",
        "target_x_m",
        "target_y_m",
        "target_radius_m",
        "reaction_time_s",
    }
    missing = required_source - set(source.columns)
    if missing:
        raise ValueError(f"Source trial data is missing columns: {sorted(missing)}")
    merged = stats.merge(
        source[list(required_source)],
        on=["session_id", "trial_id"],
        how="left",
        validate="one_to_one",
    )
    if merged[["cursor_start_x_m", "target_x_m", "target_radius_m"]].isna().any().any():
        raise ValueError("Some exported trials could not be matched to their source geometry")

    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for _, group in merged.groupby(["participant_id", "cpi"], sort=True):
        n = min(trials_per_block, len(group))
        indices = np.sort(rng.choice(len(group), size=n, replace=False))
        selected.append(group.iloc[indices].copy())
    return pd.concat(selected, ignore_index=True)


def task_from_trial(row: pd.Series, mouse_gain: float) -> dict[str, Any]:
    return {
        "window_width": 0.5313124474,
        "cursor_pos": np.array(
            [float(row["cursor_start_x_m"]), float(row["cursor_start_y_m"])]
        ),
        "cursor_vel": np.zeros(2),
        "hand_pos": np.zeros(2),
        "hand_vel": np.zeros(2),
        "target_pos": np.array([float(row["target_x_m"]), float(row["target_y_m"])]),
        "target_rad": float(row["target_radius_m"]),
        "target_spd": 0.0,
        "target_dir": 0.0,
        "target_rew": 0.0,
        "mouse_gain": mouse_gain,
        # Human hand/mouse orientation was not directly recorded. Keep the
        # simulator's trained random-orientation rule for this first check.
        "mouse_ori": None,
    }


def reaction_from_trial(row: pd.Series) -> dict[str, int]:
    reaction_ms = int(round(float(row.get("reaction_time_s", 0.2)) * 1000.0))
    return {
        FIELD.HAND_RT: reaction_ms,
        FIELD.CLICK_PRD: 0,
    }


def simulate_groups(
    selected: pd.DataFrame,
    fitted_parameters: pd.DataFrame,
    inferer: PnCInferer,
    *,
    force_true_cpi: bool,
    seed: int,
    mouse_gain: float,
) -> pd.DataFrame:
    simulator = inferer.simulator.simulator
    parameter_names = list(inferer.simulator.total_target_param)
    rows: list[pd.DataFrame] = []

    for group_number, ((participant_id, true_cpi), human_group) in enumerate(
        selected.groupby(["participant_id", "cpi"], sort=True)
    ):
        parameter_match = fitted_parameters[
            fitted_parameters["participant_id"].eq(participant_id)
            & fitted_parameters["true_cpi"].eq(float(true_cpi))
        ]
        if len(parameter_match) != 1:
            raise ValueError(
                f"Expected one fitted parameter row for {participant_id} / {true_cpi} CPI, "
                f"found {len(parameter_match)}"
            )
        parameter_row = parameter_match.iloc[0]
        parameter_w = {name: float(parameter_row[name]) for name in parameter_names}
        fitted_cpi = parameter_w["cpi"]
        if force_true_cpi:
            parameter_w["cpi"] = float(true_cpi)

        np.random.seed(seed + group_number)
        task_list = [task_from_trial(row, mouse_gain) for _, row in human_group.iterrows()]
        user_stat_list = [reaction_from_trial(row) for _, row in human_group.iterrows()]
        simulator.clear_simulation()
        simulator.fix_total_param(param_w=parameter_w)
        simulator.simulate(
            task_list=task_list,
            user_stat_list=user_stat_list,
            resimulate_max_num=0,
            verbose=False,
        )
        model_stats = simulator.get_simulation_result(return_traj=False).reset_index(drop=True)
        if len(model_stats) != len(human_group):
            raise RuntimeError("The simulator returned an unexpected number of trials")
        model_stats.insert(0, "participant_id", participant_id)
        model_stats.insert(1, "true_cpi", float(true_cpi))
        model_stats.insert(2, "trial_uid", human_group["trial_uid"].to_numpy())
        model_stats.insert(3, "fitted_cpi_before_override", fitted_cpi)
        model_stats.insert(4, "simulation_cpi", parameter_w["cpi"])
        rows.append(model_stats)
    return pd.concat(rows, ignore_index=True)


def paired_trial_table(human: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    identity = ["participant_id", "true_cpi", "trial_uid"]
    human_columns = [*identity, *METRICS]
    model_columns = [
        *identity,
        "fitted_cpi_before_override",
        "simulation_cpi",
        *METRICS,
    ]
    human_part = human.rename(columns={"cpi": "true_cpi"})[human_columns].rename(
        columns={name: f"human_{name}" for name in METRICS}
    )
    model_part = model[model_columns].rename(
        columns={name: f"model_{name}" for name in METRICS}
    )
    return human_part.merge(model_part, on=identity, how="inner", validate="one_to_one")


def target_match_table(human: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    for human_row, model_row in zip(human.itertuples(index=False), model.itertuples(index=False)):
        human_radius = float(human_row.target_radius)
        model_radius = float(getattr(model_row, "target_radius"))
        human_distance = float(human_row.target_distance)
        model_distance = float(getattr(model_row, "target_distance"))
        checks.append({
            "trial_uid": human_row.trial_uid,
            "participant_id": human_row.participant_id,
            "cpi": float(human_row.cpi),
            "human_target_radius_m": human_radius,
            "model_target_radius_m": model_radius,
            "radius_absolute_difference_m": abs(human_radius - model_radius),
            "human_target_distance_m": human_distance,
            "model_target_distance_m": model_distance,
            "distance_absolute_difference_m": abs(human_distance - model_distance),
        })
    return pd.DataFrame(checks)


def summarize_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (participant_id, cpi), group in paired.groupby(
        ["participant_id", "true_cpi"], sort=True
    ):
        for metric, metadata in METRICS.items():
            scale = float(metadata["scale"])
            for source in ("human", "model"):
                values = pd.to_numeric(group[f"{source}_{metric}"], errors="coerce").dropna()
                values = values * scale
                rows.append({
                    "participant_id": participant_id,
                    "cpi": float(cpi),
                    "source": source,
                    "metric": metric,
                    "metric_label": metadata["label"],
                    "n_trials": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "sem": (
                        float(values.std(ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    participants = list(summary["participant_id"].drop_duplicates())
    colors = plt.cm.tab10(np.linspace(0.0, 0.8, len(participants)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    for axis, (metric, metadata) in zip(axes.flat, METRICS.items()):
        for color, participant in zip(colors, participants):
            subset = summary[
                summary["participant_id"].eq(participant)
                & summary["metric"].eq(metric)
            ]
            for source, linestyle, marker in (
                ("human", "-", "o"),
                ("model", "--", "s"),
            ):
                group = subset[subset["source"].eq(source)].sort_values("cpi")
                axis.errorbar(
                    group["cpi"],
                    group["mean"],
                    yerr=group["sem"].fillna(0.0),
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2,
                    capsize=3,
                    label=f"{participant} — {source}",
                )
        axis.set_title(str(metadata["label"]))
        axis.set_xlabel("Physical CPI")
        axis.set_xticks([400, 800, 1600])
        axis.grid(alpha=0.25)
        if bool(metadata["percent"]):
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        if metric == "click_distance":
            axis.set_yscale("log")
        axis.legend(fontsize=7)
    fig.suptitle(
        "Matched-task posterior predictive check\n"
        "Solid: human | Dashed: fitted RL on identical targets",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--amortizer-model", default=DEFAULT_AMORTIZER)
    parser.add_argument("--checkpoint", type=int, default=200)
    parser.add_argument("--trials-per-block", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--mouse-gain", type=float, default=4.357877685367455)
    parser.add_argument(
        "--use-inferred-cpi",
        action="store_true",
        help="Use the inferred CPI rather than fixing the known experimental CPI.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.trials_per_block <= 0:
        raise ValueError("trials-per-block must be positive")
    inference_dir = args.inference_dir or latest_inference(DEFAULT_INFERENCE_ROOT)
    export_dir = source_export(inference_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or inference_dir / f"matched_task_simulation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    human_stats = pd.read_csv(export_dir / "human_amortizer_stats.csv")
    source_trials = load_source_trials(human_stats, DEFAULT_HUMAN_ROOT)
    selected = select_matched_trials(
        human_stats, source_trials, args.trials_per_block, args.seed
    )
    fitted = pd.read_csv(inference_dir / "human_inferred_parameters_full_trials.csv")
    fitted = fitted[fitted["model"].eq(args.amortizer_model)].copy()
    if fitted.empty:
        raise ValueError(f"No fitted parameters found for {args.amortizer_model}")

    inferer = PnCInferer(args.amortizer_model, ckpt=args.checkpoint, verbose=True)
    model_stats = simulate_groups(
        selected,
        fitted,
        inferer,
        force_true_cpi=not args.use_inferred_cpi,
        seed=args.seed,
        mouse_gain=args.mouse_gain,
    )
    paired = paired_trial_table(selected, model_stats)
    target_check = target_match_table(selected, model_stats)
    summary = summarize_paired(paired)

    paired.to_csv(output_dir / "matched_human_model_trials.csv", index=False)
    model_stats.to_csv(output_dir / "matched_model_trials.csv", index=False)
    target_check.to_csv(output_dir / "target_match_check.csv", index=False)
    summary.to_csv(output_dir / "matched_human_model_summary.csv", index=False)
    plot_summary(summary, output_dir / "matched_human_model_comparison.png")

    print(f"Amortizer parameters: {args.amortizer_model}")
    print(f"Human trials simulated: {len(selected)}")
    print(f"CPI fixed to experimental ground truth: {not args.use_inferred_cpi}")
    print(
        "Maximum target radius difference (m): "
        f"{target_check['radius_absolute_difference_m'].max():.3e}"
    )
    print(
        "Maximum target distance difference (m): "
        f"{target_check['distance_absolute_difference_m'].max():.3e}"
    )
    print(f"Results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
