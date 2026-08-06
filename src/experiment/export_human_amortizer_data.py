"""Export collected human trials in the curvature-amortizer input schema.

The experiment deliberately keeps its lossless raw CSV files.  This command
creates a smaller, analysis-ready view without modifying those source files.

Example:
    python -m src.experiment.export_human_amortizer_data
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.trajectory_features import CURVATURE_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data" / "human_experiment"

AMORTIZER_FEATURES = (
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
    *CURVATURE_FEATURE_NAMES,
)

IDENTIFIER_FIELDS = (
    "trial_uid",
    "session_id",
    "participant_id",
    "trial_id",
    "block_index",
    "cpi",
    "phase",
    "outcome",
    "mousepad_condition",
    "input_backend",
    "declared_mouse_polling_hz",
    "cpi_setting_method",
    "cpi_confirmed",
)

TRAJECTORY_FIELDS = (
    "trial_uid",
    "session_id",
    "participant_id",
    "trial_id",
    "cpi",
    "sample_index",
    "timestamp",
    "timediff",
    "cursor_pos_x",
    "cursor_pos_y",
    "target_rel_pos_x",
    "target_rel_pos_y",
)


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _session_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    return sorted(
        item
        for item in data_root.iterdir()
        if item.is_dir()
        and (item / "session.json").is_file()
        and (item / "trials.csv").is_file()
        and (item / "raw_mouse_events.csv").is_file()
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _protocol_version(session: dict[str, Any]) -> str:
    task = session.get("task") or {}
    task_config = task.get("config") or {}
    experiment = task_config.get("experiment") or {}
    return str(
        task.get("protocol_version")
        or experiment.get("protocol_version")
        or ""
    )


def _trial_selection_reason(
    row: dict[str, str],
    *,
    include_practice: bool,
    allow_unconfirmed_cpi: bool,
) -> str | None:
    if row.get("trial_stage", "task") != "task":
        return "not_task_stage"
    if not include_practice and row.get("phase") != "measured":
        return "not_measured_phase"
    if _bool(row.get("valid")) is not True:
        return "invalid_trial"
    if row.get("outcome") not in {"success", "miss"}:
        return "no_completed_click"
    if not allow_unconfirmed_cpi and _bool(row.get("cpi_confirmed")) is not True:
        return "cpi_not_confirmed"
    required = (
        "movement_time_s",
        "cursor_start_x_m",
        "cursor_start_y_m",
        "target_x_m",
        "target_y_m",
        "target_radius_m",
        "click_x_m",
        "click_y_m",
        "click_error_m",
        "path_length_m",
    )
    if any(_float(row.get(name)) is None for name in required):
        return "missing_trial_summary_value"
    if any(_float(row.get(name)) is None for name in CURVATURE_FEATURE_NAMES):
        return "missing_curvature_feature"
    return None


def _resample_movement(
    raw_rows: list[dict[str, str]],
    trial: dict[str, str],
    trajectory_hz: float,
) -> tuple[list[dict[str, float]], float] | None:
    reaction_time = _float(trial.get("reaction_time_s"))
    if reaction_time is None:
        return None

    points: list[tuple[float, float, float, float]] = []
    max_cursor_speed = 0.0
    for row in raw_rows:
        onset_ns = _float(row.get("time_from_onset_ns"))
        x = _float(row.get("cursor_x_m"))
        y = _float(row.get("cursor_y_m"))
        speed = _float(row.get("speed_m_s"))
        if speed is not None:
            max_cursor_speed = max(max_cursor_speed, speed)
        if onset_ns is None or x is None or y is None:
            continue
        onset_s = onset_ns / 1e9
        if onset_s + 1e-12 < reaction_time:
            continue
        points.append((onset_s - reaction_time, x, y, onset_s))

    if len(points) < 2:
        return None
    points.sort(key=lambda item: item[0])

    # WM_INPUT timestamps can occasionally collide.  Keep the last position at
    # each timestamp so numpy interpolation receives a strictly increasing axis.
    deduplicated: list[tuple[float, float, float, float]] = []
    for point in points:
        if deduplicated and point[0] <= deduplicated[-1][0]:
            deduplicated[-1] = point
        else:
            deduplicated.append(point)
    if len(deduplicated) < 2 or deduplicated[-1][0] <= 0:
        return None

    old_t = np.asarray([item[0] for item in deduplicated], dtype=np.float64)
    old_x = np.asarray([item[1] for item in deduplicated], dtype=np.float64)
    old_y = np.asarray([item[2] for item in deduplicated], dtype=np.float64)
    sample_count = max(2, round(float(old_t[-1]) * trajectory_hz) + 1)
    new_t = np.linspace(0.0, float(old_t[-1]), sample_count)
    new_x = np.interp(new_t, old_t, old_x)
    new_y = np.interp(new_t, old_t, old_y)
    target_x = float(trial["target_x_m"])
    target_y = float(trial["target_y_m"])

    trajectory = []
    previous_t = 0.0
    for index, (timestamp, x, y) in enumerate(zip(new_t, new_x, new_y)):
        timestamp = float(timestamp)
        trajectory.append({
            "sample_index": index,
            "timestamp": timestamp,
            "timediff": 0.0 if index == 0 else timestamp - previous_t,
            "cursor_pos_x": float(x),
            "cursor_pos_y": float(y),
            "target_rel_pos_x": target_x - float(x),
            "target_rel_pos_y": target_y - float(y),
        })
        previous_t = timestamp
    return trajectory, max_cursor_speed


def _amortizer_row(
    trial: dict[str, str],
    *,
    session_id: str,
    participant_id: str,
    hardware: dict[str, Any],
    max_cursor_speed: float,
) -> dict[str, Any]:
    trial_id = str(trial["trial_id"])
    target_x = float(trial["target_x_m"])
    target_y = float(trial["target_y_m"])
    start_x = float(trial["cursor_start_x_m"])
    start_y = float(trial["cursor_start_y_m"])
    click_x = float(trial["click_x_m"])
    click_y = float(trial["click_y_m"])
    cpi_confirmed = _bool(trial.get("cpi_confirmed")) is True
    row: dict[str, Any] = {
        "trial_uid": f"{session_id}:{trial_id}",
        "session_id": session_id,
        "participant_id": participant_id,
        "trial_id": trial_id,
        "block_index": trial.get("block_index", ""),
        "cpi": int(float(trial["cpi"])),
        "phase": trial.get("phase", ""),
        "outcome": trial.get("outcome", ""),
        "mousepad_condition": trial.get(
            "mousepad_condition", hardware.get("mousepad_condition", "unspecified")
        ),
        "input_backend": trial.get("input_backend", ""),
        "declared_mouse_polling_hz": trial.get("declared_mouse_polling_hz")
        or hardware.get("declared_mouse_polling_hz", ""),
        "cpi_setting_method": trial.get("cpi_setting_method")
        or hardware.get("cpi_setting_method", ""),
        "cpi_confirmed": cpi_confirmed,
        "movement_time": float(trial["movement_time_s"]),
        "target_hit": int(_bool(trial.get("success")) is True),
        "click_distance": float(trial["click_error_m"]),
        "target_rel_pos_x0": target_x - start_x,
        "target_rel_pos_y0": target_y - start_y,
        "click_rel_endpoint_x": target_x - click_x,
        "click_rel_endpoint_y": target_y - click_y,
        "target_radius": float(trial["target_radius_m"]),
        "target_distance": math.hypot(target_x - start_x, target_y - start_y),
        "traverse_length": float(trial["path_length_m"]),
        "max_cursor_speed": max_cursor_speed,
    }
    row.update({name: float(trial[name]) for name in CURVATURE_FEATURE_NAMES})
    return row


def export_sessions(
    data_root: Path,
    output_dir: Path,
    *,
    trajectory_hz: float = 20.0,
    include_practice: bool = False,
    allow_unconfirmed_cpi: bool = False,
    protocol_version: str | None = None,
) -> dict[str, Any]:
    if trajectory_hz <= 0:
        raise ValueError("trajectory_hz must be positive")
    session_metadata = {
        session_dir: _read_json(session_dir / "session.json")
        for session_dir in _session_dirs(data_root)
    }
    sessions = [
        session_dir
        for session_dir, session in session_metadata.items()
        if protocol_version is None
        or _protocol_version(session) == protocol_version
    ]
    if not sessions:
        suffix = (
            f" for protocol {protocol_version!r}"
            if protocol_version is not None
            else ""
        )
        raise FileNotFoundError(
            f"No complete experiment sessions found under {data_root}{suffix}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    stats_path = output_dir / "human_amortizer_stats.csv"
    trajectories_path = output_dir / "human_amortizer_trajectories.csv"
    excluded_path = output_dir / "excluded_trials.csv"
    exclusion_counts: Counter[str] = Counter()
    cpi_counts: Counter[int] = Counter()
    exported_trials = 0
    exported_samples = 0

    with (
        stats_path.open("w", newline="", encoding="utf-8") as stats_handle,
        trajectories_path.open("w", newline="", encoding="utf-8") as trajectory_handle,
        excluded_path.open("w", newline="", encoding="utf-8") as excluded_handle,
    ):
        stats_writer = csv.DictWriter(
            stats_handle, fieldnames=[*IDENTIFIER_FIELDS, *AMORTIZER_FEATURES]
        )
        trajectory_writer = csv.DictWriter(
            trajectory_handle, fieldnames=TRAJECTORY_FIELDS
        )
        excluded_writer = csv.DictWriter(
            excluded_handle, fieldnames=("session_id", "trial_id", "reason")
        )
        stats_writer.writeheader()
        trajectory_writer.writeheader()
        excluded_writer.writeheader()

        for session_dir in sessions:
            session = session_metadata[session_dir]
            session_id = str(session.get("session_id") or session_dir.name)
            participant = session.get("participant") or {}
            participant_id = str(participant.get("id") or session_id)
            hardware = session.get("hardware") or {}
            candidates: dict[str, dict[str, str]] = {}

            with (session_dir / "trials.csv").open(
                "r", newline="", encoding="utf-8"
            ) as handle:
                for trial in csv.DictReader(handle):
                    reason = _trial_selection_reason(
                        trial,
                        include_practice=include_practice,
                        allow_unconfirmed_cpi=allow_unconfirmed_cpi,
                    )
                    if reason:
                        exclusion_counts[reason] += 1
                        excluded_writer.writerow({
                            "session_id": session_id,
                            "trial_id": trial.get("trial_id", ""),
                            "reason": reason,
                        })
                    else:
                        candidates[str(trial["trial_id"])] = trial

            raw_groups: dict[str, list[dict[str, str]]] = {
                trial_id: [] for trial_id in candidates
            }
            with (session_dir / "raw_mouse_events.csv").open(
                "r", newline="", encoding="utf-8"
            ) as handle:
                for raw in csv.DictReader(handle):
                    trial_id = str(raw.get("trial_id", ""))
                    if trial_id in raw_groups:
                        raw_groups[trial_id].append(raw)

            for trial_id, trial in candidates.items():
                resampled = _resample_movement(
                    raw_groups[trial_id], trial, trajectory_hz
                )
                if resampled is None:
                    reason = "insufficient_movement_trajectory"
                    exclusion_counts[reason] += 1
                    excluded_writer.writerow({
                        "session_id": session_id,
                        "trial_id": trial_id,
                        "reason": reason,
                    })
                    continue
                trajectory, max_cursor_speed = resampled
                stat = _amortizer_row(
                    trial,
                    session_id=session_id,
                    participant_id=participant_id,
                    hardware=hardware,
                    max_cursor_speed=max_cursor_speed,
                )
                stats_writer.writerow(stat)
                for sample in trajectory:
                    trajectory_writer.writerow({
                        "trial_uid": stat["trial_uid"],
                        "session_id": session_id,
                        "participant_id": participant_id,
                        "trial_id": trial_id,
                        "cpi": stat["cpi"],
                        **sample,
                    })
                exported_trials += 1
                exported_samples += len(trajectory)
                cpi_counts[int(stat["cpi"])] += 1

    manifest = {
        "schema_version": 1,
        "created_local": datetime.now().astimezone().isoformat(),
        "source_data_root": str(data_root.resolve()),
        "session_count": len(sessions),
        "source_sessions": [session_dir.name for session_dir in sessions],
        "protocol_version_filter": protocol_version,
        "exported_trial_count": exported_trials,
        "exported_trajectory_sample_count": exported_samples,
        "trajectory_hz": trajectory_hz,
        "trajectory_reaction_period_removed": True,
        "filters": {
            "include_practice": include_practice,
            "allow_unconfirmed_cpi": allow_unconfirmed_cpi,
            "protocol_version": protocol_version,
            "valid_task_trials_only": True,
            "completed_clicks_only": True,
        },
        "trials_per_cpi": {str(key): value for key, value in sorted(cpi_counts.items())},
        "excluded_by_reason": dict(sorted(exclusion_counts.items())),
        "amortizer_feature_order": list(AMORTIZER_FEATURES),
        "files": {
            "statistics": stats_path.name,
            "trajectories": trajectories_path.name,
            "excluded_trials": excluded_path.name,
        },
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Human amortizer export\n\n"
        "`human_amortizer_stats.csv` contains one row per retained trial and the "
        "21 features used by the curvature amortizer.\n\n"
        "`human_amortizer_trajectories.csv` contains movement-only trajectories "
        f"resampled at {trajectory_hz:g} Hz. Group rows by `trial_uid` before "
        "passing them to `PnCInferer.process_data`.\n\n"
        "The CPI column is experimental ground truth confirmed at the start of "
        "each block; it is retained for evaluation and must not be passed as an "
        "amortizer input feature.\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trajectory-hz", type=float, default=20.0)
    parser.add_argument("--include-practice", action="store_true")
    parser.add_argument("--allow-unconfirmed-cpi", action="store_true")
    parser.add_argument(
        "--protocol-version",
        help="Export only sessions whose recorded protocol version matches exactly.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.data_root / "amortizer_ready" / timestamp
    manifest = export_sessions(
        args.data_root,
        output_dir,
        trajectory_hz=args.trajectory_hz,
        include_practice=args.include_practice,
        allow_unconfirmed_cpi=args.allow_unconfirmed_cpi,
        protocol_version=args.protocol_version,
    )
    print(f"Exported {manifest['exported_trial_count']} trials to {output_dir.resolve()}")
    print(f"Trials per CPI: {manifest['trials_per_cpi']}")
    if manifest["excluded_by_reason"]:
        print(f"Excluded: {manifest['excluded_by_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
