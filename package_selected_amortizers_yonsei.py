#!/usr/bin/env python3
"""Package the latest completed 20 Hz trajectory and CLEAN7 amortizers."""

from __future__ import annotations

import json
import os
import re
import sys
import tarfile
from pathlib import Path

import yaml


ROOT = Path(os.environ.get("BUNDLE_DIR", Path(__file__).resolve().parent)).resolve()
MODELS_ROOT = ROOT / "data" / "amortizer_cond" / "models"
MIN_ITERATION = int(os.environ.get("MIN_ITERATION", "200"))
ITER_RE = re.compile(r"iter(\d+)\.pt$")

CURVATURE_FEATURES = {
    "cursor_total_abs_turn_rad",
    "cursor_net_turn_rad",
    "cursor_mean_abs_turn_rad",
    "cursor_p90_abs_turn_rad",
    "cursor_turn_energy_rad2",
    "cursor_max_deviation_normalized",
    "cursor_path_excess_ratio",
    "cursor_early_mean_abs_turn_rad",
    "cursor_middle_mean_abs_turn_rad",
    "cursor_late_mean_abs_turn_rad",
}


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value if isinstance(value, dict) else {}


def latest_iteration(model_dir: Path) -> int:
    iterations = []
    for checkpoint in (model_dir / "pts").glob("iter*.pt"):
        match = ITER_RE.match(checkpoint.name)
        if match:
            iterations.append(int(match.group(1)))
    return max(iterations, default=0)


def inspect_model(model_dir: Path) -> dict:
    design = load_yaml(model_dir / "shared_fit_design.yaml")
    simulator = load_yaml(model_dir / "simulator_setting.yaml")
    normalization = load_yaml(model_dir / "data_normalization_setting.yaml")

    stat = normalization.get("stat", {})
    trajectory = normalization.get("traj", {})
    stat_features = stat.get("list", []) or []
    retained_curvature = [
        feature for feature in stat_features if feature in CURVATURE_FEATURES
    ]

    trajectory_hz = design.get("trajectory_hz", simulator.get("traj_downsample"))
    variant = design.get("variant")
    ignore_trajectory = bool(trajectory.get("ignore_traj", False))
    feature_profile = design.get("feature_profile")
    if not feature_profile:
        feature_profile = "clean7" if len(retained_curvature) == 7 else "full"

    return {
        "name": model_dir.name,
        "path": str(model_dir),
        "variant": variant,
        "feature_profile": feature_profile,
        "trajectory_hz": trajectory_hz,
        "ignore_trajectory": ignore_trajectory,
        "retained_curvature_features": retained_curvature,
        "retained_curvature_count": len(retained_curvature),
        "latest_iteration": latest_iteration(model_dir),
        "rl_model": simulator.get("model_name"),
        "rl_checkpoint": simulator.get("ckpt"),
        "modified_time": model_dir.stat().st_mtime,
    }


def completed(candidates: list[dict]) -> list[dict]:
    return [item for item in candidates if item["latest_iteration"] >= MIN_ITERATION]


def choose(kind: str, candidates: list[dict], override: str | None) -> dict:
    if override:
        selected = next((item for item in candidates if item["name"] == override), None)
        if selected is None:
            raise SystemExit(f"Requested {kind} model does not exist: {override}")
        if selected["latest_iteration"] < MIN_ITERATION:
            raise SystemExit(
                f"Requested {kind} model is incomplete: {override} "
                f"(latest iteration {selected['latest_iteration']}, "
                f"required {MIN_ITERATION})"
            )
        return selected

    eligible = completed(candidates)
    if not eligible:
        print(f"No completed {kind} model reached iteration {MIN_ITERATION}.", file=sys.stderr)
        if candidates:
            print("Candidates found:", file=sys.stderr)
            for item in sorted(candidates, key=lambda x: x["modified_time"], reverse=True):
                print(
                    f"  {item['name']} (iter {item['latest_iteration']}, "
                    f"RL {item['rl_model']})",
                    file=sys.stderr,
                )
        raise SystemExit(1)
    return max(eligible, key=lambda item: item["modified_time"])


def archive_model(selected: dict, archive_name: str, label: str) -> Path:
    archive_path = ROOT / archive_name
    model_dir = Path(selected["path"])
    manifest = {
        "archive_label": label,
        "minimum_required_iteration": MIN_ITERATION,
        **{key: value for key, value in selected.items() if key != "modified_time"},
    }
    manifest_path = model_dir / "yonsei_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(
                model_dir,
                arcname=str(model_dir.relative_to(ROOT)).replace(os.sep, "/"),
            )
    finally:
        manifest_path.unlink(missing_ok=True)
    return archive_path


def main() -> int:
    if not MODELS_ROOT.is_dir():
        raise SystemExit(f"Model directory not found: {MODELS_ROOT}")

    models = [inspect_model(path) for path in MODELS_ROOT.iterdir() if path.is_dir()]

    trajectory_candidates = [
        item
        for item in models
        if item["variant"] == "trajectory"
        and not item["ignore_trajectory"]
        and int(item["trajectory_hz"] or 0) == 20
    ]
    clean7_candidates = [
        item
        for item in models
        if item["variant"] == "summary-only"
        and item["ignore_trajectory"]
        and item["feature_profile"] == "clean7"
        and item["retained_curvature_count"] == 7
    ]

    trajectory = choose(
        "20 Hz trajectory",
        trajectory_candidates,
        os.environ.get("TRAJ_MODEL"),
    )
    clean7 = choose("CLEAN7 summary-only", clean7_candidates, os.environ.get("CLEAN7_MODEL"))

    if trajectory["name"] == clean7["name"]:
        raise SystemExit("Trajectory and CLEAN7 selections unexpectedly refer to the same model")

    trajectory_archive = archive_model(
        trajectory,
        "amortizer_20hz_trajectory_complete.tar.gz",
        "20_hz_trajectory",
    )
    clean7_archive = archive_model(
        clean7,
        "amortizer_clean7_complete.tar.gz",
        "clean7_summary_only",
    )

    export = {
        "trajectory": trajectory,
        "clean7": clean7,
        "archives": {
            "trajectory": trajectory_archive.name,
            "clean7": clean7_archive.name,
        },
    }
    (ROOT / "selected_amortizers_export.json").write_text(
        json.dumps(export, indent=2) + "\n", encoding="utf-8"
    )

    print("Selected completed amortizers:")
    print(
        f"  trajectory: {trajectory['name']} "
        f"(iter {trajectory['latest_iteration']}, {trajectory['trajectory_hz']} Hz, "
        f"RL {trajectory['rl_model']})"
    )
    print(
        f"  clean7:     {clean7['name']} "
        f"(iter {clean7['latest_iteration']}, "
        f"RL {clean7['rl_model']})"
    )
    print("Created:")
    print(f"  {trajectory_archive} ({trajectory_archive.stat().st_size / 1024**2:.1f} MiB)")
    print(f"  {clean7_archive} ({clean7_archive.stat().st_size / 1024**2:.1f} MiB)")
    print(f"  {ROOT / 'selected_amortizers_export.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
