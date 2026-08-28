"""Compare virtual-cursor and native-Windows-cursor pilot sessions.

Both conditions are expected to use the Windows Raw Input backend. The virtual
condition reconstructs its cursor from counts; the corrected native condition
uses mapped GetCursorPos positions for the visible trajectory and hit test while
saving the count reconstruction separately.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = (
    "success",
    "reaction_time_s",
    "movement_time_s",
    "path_length_m",
    "straight_line_efficiency",
    "raw_max_speed_m_s",
    "cursor_turn_energy_rad2",
    "cursor_early_mean_abs_turn_rad",
    "cursor_late_mean_abs_turn_rad",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_session(path: Path, expected_mode: str) -> tuple[pd.DataFrame, dict]:
    metadata_path = path / "session.json"
    trials_path = path / "trials.csv"
    if not metadata_path.is_file() or not trials_path.is_file():
        raise FileNotFoundError(f"Session files are missing in {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mode = metadata.get("hardware", {}).get("cursor_presentation_mode")
    if mode != expected_mode:
        raise ValueError(f"Expected {expected_mode!r} in {path}, found {mode!r}")
    if metadata.get("input_backend") != "windows_raw_input":
        raise ValueError(f"{path} did not use Windows Raw Input")
    frame = pd.read_csv(trials_path)
    required = {"phase", "trial_stage", "valid", "cpi", *FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {trials_path}: {missing}")
    selected = frame[
        frame["phase"].eq("measured")
        & frame["trial_stage"].eq("task")
        & _as_bool(frame["valid"])
    ].copy()
    selected["cursor_presentation_mode"] = expected_mode
    selected["success"] = _as_bool(selected["success"]).astype(float)
    for feature in FEATURES[1:]:
        selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
    selected["cpi"] = pd.to_numeric(selected["cpi"], errors="raise")
    return selected, metadata


def feature_statistics(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (mode, cpi), group in combined.groupby(
        ["cursor_presentation_mode", "cpi"], sort=True
    ):
        for feature in FEATURES:
            values = group[feature].dropna().to_numpy(float)
            mode_all = combined.loc[
                combined["cursor_presentation_mode"].eq(mode), ["cpi", feature]
            ].dropna()
            slope = (
                float(np.polyfit(mode_all["cpi"], mode_all[feature], 1)[0] * 1000.0)
                if len(mode_all) >= 2 and mode_all["cpi"].nunique() >= 2
                else np.nan
            )
            rows.append(
                {
                    "cursor_presentation_mode": mode,
                    "cpi": int(cpi),
                    "feature": feature,
                    "n": int(values.size),
                    "mean": float(np.mean(values)) if values.size else np.nan,
                    "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                    "cpi_slope_per_1000_cpi": slope,
                }
            )
    return pd.DataFrame(rows)


def mode_differences(statistics: pd.DataFrame) -> pd.DataFrame:
    virtual = statistics[
        statistics["cursor_presentation_mode"].eq("virtual_raw_input")
    ].set_index(["cpi", "feature"])
    native = statistics[
        statistics["cursor_presentation_mode"].eq("native_windows")
    ].set_index(["cpi", "feature"])
    rows: list[dict] = []
    for key in virtual.index.intersection(native.index):
        cpi, feature = key
        left, right = virtual.loc[key], native.loc[key]
        pooled_denominator = max(int(left["n"] + right["n"] - 2), 1)
        pooled_variance = (
            max(int(left["n"] - 1), 0) * float(left["sd"] ** 2)
            + max(int(right["n"] - 1), 0) * float(right["sd"] ** 2)
        ) / pooled_denominator
        pooled_sd = float(np.sqrt(max(pooled_variance, 0.0)))
        difference = float(right["mean"] - left["mean"])
        rows.append(
            {
                "cpi": int(cpi),
                "feature": feature,
                "virtual_mean": float(left["mean"]),
                "native_mean": float(right["mean"]),
                "native_minus_virtual": difference,
                "standardized_mean_difference": (
                    difference / pooled_sd if pooled_sd > 0 else np.nan
                ),
                "virtual_cpi_slope_per_1000_cpi": float(
                    left["cpi_slope_per_1000_cpi"]
                ),
                "native_cpi_slope_per_1000_cpi": float(
                    right["cpi_slope_per_1000_cpi"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature", "cpi"])


def plot_statistics(statistics: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    colors = {"virtual_raw_input": "#2369bd", "native_windows": "#e36b2c"}
    labels = {"virtual_raw_input": "Virtual Raw Input", "native_windows": "Native Windows"}
    for axis, feature in zip(axes.flat, FEATURES):
        feature_rows = statistics[statistics["feature"].eq(feature)]
        for mode in ("virtual_raw_input", "native_windows"):
            line = feature_rows[
                feature_rows["cursor_presentation_mode"].eq(mode)
            ].sort_values("cpi")
            axis.errorbar(
                line["cpi"], line["mean"], yerr=line["sd"].fillna(0),
                marker="o", capsize=3, linewidth=1.8, color=colors[mode],
                label=labels[mode],
            )
        axis.set_title(feature.replace("_", " "))
        axis.set_xticks([400, 800, 1600])
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Cursor-presentation A/B pilot: measured human trials", fontsize=16)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_summary(
    output: Path,
    virtual: pd.DataFrame,
    native: pd.DataFrame,
    differences: pd.DataFrame,
) -> None:
    largest = differences.dropna(subset=["standardized_mean_difference"]).copy()
    largest["absolute_smd"] = largest["standardized_mean_difference"].abs()
    largest = largest.sort_values("absolute_smd", ascending=False).head(8)
    alignment_field = (
        "native_raw_alignment_p95_m"
        if "native_raw_alignment_p95_m" in native.columns
        and native["native_raw_alignment_p95_m"].notna().any()
        else "native_alignment_p95_m"
    )
    alignment = native.get(alignment_field, pd.Series(dtype=float)).dropna()
    lines = [
        "# Cursor-presentation A/B pilot",
        "",
        f"- Virtual measured trials: {len(virtual)}",
        f"- Native measured trials: {len(native)}",
        "- Both conditions save the same Windows Raw Input counts.",
        "- The corrected native condition uses mapped GetCursorPos coordinates for its visible trajectory and hit test.",
    ]
    if len(alignment):
        lines.append(
            f"- Median trial-level native/count-reconstructed P95 alignment error: "
            f"{1000 * float(alignment.median()):.2f} mm."
        )
    lines.extend(["", "## Largest condition differences", ""])
    if largest.empty:
        lines.append("No finite standardized differences were available.")
    else:
        lines.append("| CPI | Feature | Native - virtual | Standardized difference |")
        lines.append("|---:|---|---:|---:|")
        for row in largest.itertuples():
            lines.append(
                f"| {row.cpi} | {row.feature} | {row.native_minus_virtual:.6g} | "
                f"{row.standardized_mean_difference:.3f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation: absolute standardized differences around 0.2 are small, "
            "0.5 moderate, and 0.8 large. A large native alignment error means the "
            "visible Windows cursor and the Raw Input trajectory did not stay "
            "sufficiently aligned, so that native condition should not replace the "
            "controlled cursor without further calibration.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--virtual-session", type=Path, required=True)
    parser.add_argument("--native-session", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir or (
        Path("data/diagnostics")
        / f"cursor_presentation_ab_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    virtual, _ = load_session(args.virtual_session, "virtual_raw_input")
    native, _ = load_session(args.native_session, "native_windows")
    combined = pd.concat([virtual, native], ignore_index=True)
    statistics = feature_statistics(combined)
    differences = mode_differences(statistics)
    statistics.to_csv(output_dir / "cursor_ab_feature_statistics.csv", index=False)
    differences.to_csv(output_dir / "cursor_ab_mode_differences.csv", index=False)
    plot_statistics(statistics, output_dir / "cursor_ab_feature_trends.png")
    write_summary(output_dir / "cursor_ab_summary.md", virtual, native, differences)
    print(f"A/B comparison complete: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
