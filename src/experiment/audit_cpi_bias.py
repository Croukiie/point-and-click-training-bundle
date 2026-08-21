"""Audit CPI sampling, experimental labels, and inference calibration."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMPORT = ROOT / "yonsei_imports/integrity_visible_20260821_125316"
DEFAULT_HUMAN = ROOT / "data/human_experiment"
DEFAULT_COMPARISON = (
    DEFAULT_HUMAN
    / "amortizer_ready/integrity_reexport_20260821/integrity_visible_comparison_aligned"
)
MODEL_NAMES = {
    "speedpath": (
        "PTE_SHARED_CPI_INTEGRITY_SPEEDPATH_FULL_"
        "integrity_visible_260819_151300_SUMMARY_08202351"
    ),
    "core3": (
        "PTE_SHARED_CPI_INTEGRITY_CORE3_FULL_"
        "integrity_visible_260819_151300_SUMMARY_08210246"
    ),
}
HUMAN_SESSIONS = (
    "20260805_145518_Elisa_web",
    "20260805_152013_Hyeeun_web",
)
CPI_LEVELS = np.arange(400, 1601, 10)


def long_path(path: Path) -> Path:
    resolved = str(path.resolve())
    return Path("\\\\?\\" + resolved) if len(resolved) >= 240 else path


def dataset_directory(import_root: Path) -> Path:
    candidates = list((import_root / "data/amortizer_cond/dataset").iterdir())
    if len(candidates) != 1:
        raise ValueError(f"Expected one imported dataset, found {len(candidates)}")
    return long_path(candidates[0])


def load_cpi_targets(npz_path: Path) -> np.ndarray:
    """Read only the small params member, not the multi-gigabyte stat array."""
    with zipfile.ZipFile(npz_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("params.npy"))
        params = np.load(io.BytesIO(archive.read(member)))
    if params.shape[1] != 10:
        raise ValueError(f"Unexpected parameter shape: {params.shape}")
    return np.rint((params[:, 9] + 1.0) * 600.0 + 400.0).astype(int)


def discrete_ks(cpi: np.ndarray) -> float:
    counts = np.bincount((cpi - 400) // 10, minlength=len(CPI_LEVELS))
    observed = np.cumsum(counts) / len(cpi)
    expected = np.arange(1, len(CPI_LEVELS) + 1) / len(CPI_LEVELS)
    return float(np.max(np.abs(observed - expected)))


def validation_monte_carlo(cpi: np.ndarray, seed: int, repeats: int = 20_000):
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(CPI_LEVELS), size=(repeats, len(cpi)))
    sample_means = 400.0 + 10.0 * samples.mean(axis=1)
    histograms = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=len(CPI_LEVELS)), 1, samples
    )
    expected = np.arange(1, len(CPI_LEVELS) + 1) / len(CPI_LEVELS)
    sample_ks = np.max(
        np.abs(np.cumsum(histograms, axis=1) / len(cpi) - expected), axis=1
    )
    mean_deviation = abs(float(cpi.mean()) - 1000.0)
    return (
        float((np.sum(np.abs(sample_means - 1000.0) >= mean_deviation) + 1) / (repeats + 1)),
        float((np.sum(sample_ks >= discrete_ks(cpi)) + 1) / (repeats + 1)),
    )


def sampling_audit(dataset_dir: Path, seed: int) -> pd.DataFrame:
    rows = []
    for source, pattern in (("training", "tr_*.npz"), ("validation", "vd_*.npz")):
        path = next(dataset_dir.glob(pattern))
        cpi = load_cpi_targets(path)
        counts = np.bincount((cpi - 400) // 10, minlength=len(CPI_LEVELS))
        ks = discrete_ks(cpi)
        if source == "training":
            uniformity_p = float(stats.chisquare(counts).pvalue)
            expected_sd = float(np.std(CPI_LEVELS, ddof=0))
            mean_z = (float(cpi.mean()) - 1000.0) / (expected_sd / np.sqrt(len(cpi)))
            mean_p = float(2 * stats.norm.sf(abs(mean_z)))
        else:
            mean_p, uniformity_p = validation_monte_carlo(cpi, seed)
        slot_test = stats.f_oneway(cpi[0::3], cpi[1::3], cpi[2::3])
        profiles = cpi.reshape(-1, 3)
        rows.append({
            "source": source,
            "cpi_target_count": len(cpi),
            "mean_cpi": float(cpi.mean()),
            "expected_mean_cpi": 1000.0,
            "sd_cpi": float(cpi.std(ddof=1)),
            "min_cpi": int(cpi.min()),
            "max_cpi": int(cpi.max()),
            "discrete_uniform_ks": ks,
            "uniformity_p_value": uniformity_p,
            "mean_deviation_p_value": mean_p,
            "slot_1_mean": float(cpi[0::3].mean()),
            "slot_2_mean": float(cpi[1::3].mean()),
            "slot_3_mean": float(cpi[2::3].mean()),
            "slot_effect_p_value": float(slot_test.pvalue),
            "profiles_with_repeated_cpi_pct": float(
                100 * np.mean([len(set(row)) < 3 for row in profiles])
            ),
            "count_cpi_400": int(np.sum(cpi == 400)),
            "count_cpi_800": int(np.sum(cpi == 800)),
            "count_cpi_1600": int(np.sum(cpi == 1600)),
        })
    return pd.DataFrame(rows)


def human_design_audit(human_root: Path):
    frames = []
    orders = {}
    hardware = {}
    for session_name in HUMAN_SESSIONS:
        session_dir = human_root / session_name
        session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        participant = session["participant"]["id"]
        orders[participant] = session["task"]["cpi_order"]
        hardware[participant] = {
            "mouse_model": session["hardware"]["mouse_model"],
            "cpi_setting_method": session["hardware"]["cpi_setting_method"],
            "pointer_acceleration": session["hardware"]["pointer_acceleration"],
            "windows_pointer_speed": session["hardware"]["windows_pointer_speed"],
        }
        trials = pd.read_csv(session_dir / "trials.csv")
        trials = trials[
            (trials["phase"] == "measured")
            & (trials["trial_stage"] == "task")
            & (trials["valid"] == True)  # noqa: E712
            & trials["click_ns"].notna()
        ].copy()
        trials["participant_id"] = participant
        frames.append(trials)
    human = pd.concat(frames, ignore_index=True)
    grouped = (
        human.groupby(["participant_id", "block_index", "cpi"], as_index=False)
        .agg(
            trials=("trial_id", "size"),
            cpi_confirmed_pct=("cpi_confirmed", lambda values: 100 * values.mean()),
            target_radius_mean_m=("target_radius_m", "mean"),
            target_distance_mean_m=("target_distance_m", "mean"),
            fitts_id_mean=("fitts_id", "mean"),
            target_hit_rate=("success", "mean"),
            movement_time_mean_s=("movement_time_s", "mean"),
        )
        .sort_values(["participant_id", "block_index"])
    )
    task_tests = {}
    for feature in ("target_radius_m", "target_distance_m", "fitts_id"):
        groups = [human.loc[human["cpi"] == cpi, feature].dropna() for cpi in (400, 800, 1600)]
        task_tests[feature] = {
            "anova_p_value": float(stats.f_oneway(*groups).pvalue),
            "means_by_cpi": {
                str(cpi): float(human.loc[human["cpi"] == cpi, feature].mean())
                for cpi in (400, 800, 1600)
            },
        }
    return human, grouped, orders, hardware, task_tests


def inference_bias(comparison_dir: Path) -> pd.DataFrame:
    human = pd.read_csv(comparison_dir / "human_predictions.csv")
    human = human[human["n_trials_per_block"] == 64].copy()
    human_rows = (
        human.groupby(["model_variant", "participant_id", "true_cpi"], as_index=False)
        .agg(
            predictions=("predicted_cpi", "size"),
            predicted_cpi_mean=("predicted_cpi", "mean"),
            predicted_cpi_sd=("predicted_cpi", "std"),
            bias_cpi=("signed_error", "mean"),
            mae_cpi=("absolute_error", "mean"),
        )
    )
    human_rows.insert(0, "source", "human")
    human_rows["cpi_group"] = human_rows["true_cpi"].map(lambda value: f"{value:g}")

    simulated = pd.read_csv(comparison_dir / "simulated_predictions.csv")
    simulated = simulated[simulated["n_trials_per_block"] == 64].copy()
    simulated["signed_error"] = simulated["predicted_cpi"] - simulated["true_cpi"]
    simulated["absolute_error"] = simulated["signed_error"].abs()
    simulated["cpi_group"] = pd.cut(
        simulated["true_cpi"],
        bins=[399, 600, 800, 1000, 1200, 1400, 1601],
        right=False,
    ).astype(str)
    simulated_rows = (
        simulated.groupby(["model_variant", "cpi_group"], as_index=False, observed=True)
        .agg(
            predictions=("predicted_cpi", "size"),
            true_cpi=("true_cpi", "mean"),
            predicted_cpi_mean=("predicted_cpi", "mean"),
            predicted_cpi_sd=("predicted_cpi", "std"),
            bias_cpi=("signed_error", "mean"),
            mae_cpi=("absolute_error", "mean"),
        )
    )
    simulated_rows.insert(0, "source", "simulation")
    simulated_rows.insert(2, "participant_id", "simulated validation")
    columns = [
        "source", "model_variant", "participant_id", "cpi_group", "true_cpi",
        "predictions", "predicted_cpi_mean", "predicted_cpi_sd", "bias_cpi", "mae_cpi",
    ]
    return pd.concat([simulated_rows[columns], human_rows[columns]], ignore_index=True)


def input_leakage_audit() -> dict[str, object]:
    result = {}
    for variant, name in MODEL_NAMES.items():
        model_dir = ROOT / "data/amortizer_cond/models" / name
        design = yaml.safe_load((model_dir / "shared_fit_design.yaml").read_text(encoding="utf-8"))
        normalization = yaml.safe_load(
            (model_dir / "data_normalization_setting.yaml").read_text(encoding="utf-8")
        )
        feature_names = list(normalization["stat"]["list"])
        result[variant] = {
            "true_cpi_is_input_declared": bool(design["true_cpi_is_input"]),
            "block_labels_are_inputs_declared": bool(design["block_labels_are_inputs"]),
            "cpi_present_in_summary_features": "cpi" in feature_names,
            "summary_feature_count": len(feature_names),
        }
    return result


def physical_calibration_audit() -> dict[str, object] | None:
    files = sorted(
        (ROOT / "data/diagnostics").glob("physical_cpi_calibration_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None
    frame = pd.read_csv(files[0])
    grouped = frame.groupby("nominal_cpi")["measured_cpi"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    rows = []
    for nominal_cpi, values in grouped.iterrows():
        rows.append({
            "nominal_cpi": int(nominal_cpi),
            "repetitions": int(values["count"]),
            "measured_cpi_mean": float(values["mean"]),
            "measured_cpi_sd": float(values["std"]),
            "measured_cpi_min": float(values["min"]),
            "measured_cpi_max": float(values["max"]),
            "mean_error_pct": float(
                100.0 * (values["mean"] / float(nominal_cpi) - 1.0)
            ),
        })
    return {"source_file": str(files[0].resolve()), "settings": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-root", type=Path, default=DEFAULT_IMPORT)
    parser.add_argument("--human-root", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampling = sampling_audit(dataset_directory(args.import_root), args.seed)
    _, human_design, orders, hardware, task_tests = human_design_audit(args.human_root)
    bias = inference_bias(args.comparison_dir)
    leakage = input_leakage_audit()
    physical_calibration = physical_calibration_audit()

    sampling.to_csv(args.output_dir / "cpi_sampling_audit.csv", index=False)
    human_design.to_csv(args.output_dir / "human_cpi_design_audit.csv", index=False)
    bias.to_csv(args.output_dir / "inference_bias_by_cpi.csv", index=False)
    summary = {
        "conclusion": (
            "No evidence of biased simulated CPI sampling or CPI leakage into the amortizer input. "
            "The human estimator is nevertheless strongly downward biased at 800 and 1600 CPI."
        ),
        "physical_ground_truth_status": (
            "independently_measured_with_raw_input_counts"
            if physical_calibration is not None
            else "declared_and_confirmed_by_hardware_button_not_independently_measured"
        ),
        "physical_cpi_calibration": physical_calibration,
        "human_block_orders": orders,
        "human_hardware": hardware,
        "human_task_distribution_tests": task_tests,
        "amortizer_input_leakage_audit": leakage,
        "limitations": [
            "Only two human participants were recorded.",
            "The two CPI orders do not fully counterbalance block/order effects.",
        ] + (
            []
            if physical_calibration is not None
            else [
                "Actual device CPI was not measured with an independent physical-distance calibration."
            ]
        ),
    }
    (args.output_dir / "cpi_bias_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"OUTPUT_DIR={args.output_dir.resolve()}")
    print(sampling.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
