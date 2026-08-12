"""Quality-check summaries for human point-and-click experiment sessions.

Example:
    python -m src.experiment.quality_check_human_data

Outputs are written under data/human_experiment/qc_reports/<timestamp>/ by default.
The script uses only the Python standard library so it can run in the project venv
without extra packages.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "data" / "human_experiment"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(vals) if vals else None


def _median(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(vals) if vals else None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _session_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    sessions = []
    for item in data_root.iterdir():
        if not item.is_dir():
            continue
        if (item / "session.json").exists() and (item / "trials.csv").exists():
            sessions.append(item)
    return sorted(sessions)


@dataclass
class TrialGroupSummary:
    session_id: str
    participant_id: str
    phase: str
    cpi: str
    n_trials: int
    success: int
    miss: int
    timeout: int
    invalid: int
    success_rate: float | None
    miss_rate: float | None
    timeout_rate: float | None
    mean_movement_time_s: float | None
    median_movement_time_s: float | None
    mean_total_time_s: float | None
    mean_click_error_m: float | None
    median_click_error_m: float | None
    mean_click_error_over_radius: float | None
    mean_min_distance_m: float | None
    mean_path_length_m: float | None
    mean_straight_line_efficiency: float | None
    mean_raw_event_rate_hz: float | None
    mean_raw_event_count: float | None
    late_frame_count: int
    focus_lost_count: int
    fullscreen_false_count: int

    def to_row(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "phase": self.phase,
            "cpi": self.cpi,
            "n_trials": self.n_trials,
            "success": self.success,
            "miss": self.miss,
            "timeout": self.timeout,
            "invalid": self.invalid,
            "success_rate": self.success_rate,
            "miss_rate": self.miss_rate,
            "timeout_rate": self.timeout_rate,
            "mean_movement_time_s": self.mean_movement_time_s,
            "median_movement_time_s": self.median_movement_time_s,
            "mean_total_time_s": self.mean_total_time_s,
            "mean_click_error_m": self.mean_click_error_m,
            "median_click_error_m": self.median_click_error_m,
            "mean_click_error_over_radius": self.mean_click_error_over_radius,
            "mean_min_distance_m": self.mean_min_distance_m,
            "mean_path_length_m": self.mean_path_length_m,
            "mean_straight_line_efficiency": self.mean_straight_line_efficiency,
            "mean_raw_event_rate_hz": self.mean_raw_event_rate_hz,
            "mean_raw_event_count": self.mean_raw_event_count,
            "late_frame_count": self.late_frame_count,
            "focus_lost_count": self.focus_lost_count,
            "fullscreen_false_count": self.fullscreen_false_count,
        }


@dataclass
class SessionSummary:
    session_id: str
    participant_id: str
    input_backend: str
    dominant_hand: str
    age: Any
    n_trials: int
    n_practice: int
    n_measured: int
    measured_success_rate: float | None
    measured_miss: int
    measured_timeout: int
    invalid: int
    focus_lost_count: int
    fullscreen_false_count: int
    cpi_values: str
    display_resolution: str
    display_size_cm: str
    browser_estimated_resolution: str
    fixed_profile_warning: str
    session_complete: bool

    def to_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RawCpiSummary:
    session_id: str
    participant_id: str
    phase: str
    cpi: str
    moving_raw_events: int
    median_count_magnitude: float | None
    mean_count_magnitude: float | None
    mean_cursor_delta_mm: float | None
    median_cursor_delta_mm: float | None
    mean_raw_delta_mm: float | None
    median_dt_ms: float | None
    max_dt_ms: float | None

    def to_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _participant_id(session: dict[str, Any], session_dir: Path) -> str:
    participant = session.get("participant") or {}
    return str(participant.get("id") or session.get("session_id") or session_dir.name)


def _event_counts(session_dir: Path) -> dict[str, int]:
    events_path = session_dir / "experiment_events.jsonl"
    counts: dict[str, int] = {}
    if not events_path.exists():
        return counts
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                counts["json_decode_error"] = counts.get("json_decode_error", 0) + 1
                continue
            name = str(event.get("event") or event.get("type") or "unknown")
            counts[name] = counts.get(name, 0) + 1
    return counts


def _summarize_trial_group(
    rows: list[dict[str, str]],
    session_id: str,
    participant_id: str,
    phase: str,
    cpi: str,
) -> TrialGroupSummary:
    n = len(rows)
    outcomes = [row.get("outcome", "") for row in rows]
    success = outcomes.count("success")
    miss = outcomes.count("miss")
    timeout = outcomes.count("timeout")
    invalid = sum(_bool(row.get("valid")) is False for row in rows)
    focus_lost = sum(_bool(row.get("focus_lost")) is True for row in rows)
    fullscreen_false = sum(_bool(row.get("client_fullscreen")) is False for row in rows)
    late_frames = sum(int(_float(row.get("late_frame_count")) or 0) for row in rows)

    click_error_over_radius = []
    for row in rows:
        error = _float(row.get("click_error_m"))
        radius = _float(row.get("target_radius_m"))
        if error is not None and radius and radius > 0:
            click_error_over_radius.append(error / radius)

    return TrialGroupSummary(
        session_id=session_id,
        participant_id=participant_id,
        phase=phase,
        cpi=cpi,
        n_trials=n,
        success=success,
        miss=miss,
        timeout=timeout,
        invalid=invalid,
        success_rate=_rate(success, n),
        miss_rate=_rate(miss, n),
        timeout_rate=_rate(timeout, n),
        mean_movement_time_s=_mean(_float(row.get("movement_time_s")) for row in rows),
        median_movement_time_s=_median(_float(row.get("movement_time_s")) for row in rows),
        mean_total_time_s=_mean(_float(row.get("total_time_s")) for row in rows),
        mean_click_error_m=_mean(_float(row.get("click_error_m")) for row in rows),
        median_click_error_m=_median(_float(row.get("click_error_m")) for row in rows),
        mean_click_error_over_radius=_mean(click_error_over_radius),
        mean_min_distance_m=_mean(_float(row.get("minimum_target_distance_m")) for row in rows),
        mean_path_length_m=_mean(_float(row.get("path_length_m")) for row in rows),
        mean_straight_line_efficiency=_mean(_float(row.get("straight_line_efficiency")) for row in rows),
        mean_raw_event_rate_hz=_mean(_float(row.get("raw_event_rate_hz")) for row in rows),
        mean_raw_event_count=_mean(_float(row.get("raw_event_count")) for row in rows),
        late_frame_count=late_frames,
        focus_lost_count=focus_lost,
        fullscreen_false_count=fullscreen_false,
    )


def _group_trial_rows(
    rows: list[dict[str, str]],
    session_id: str,
    participant_id: str,
) -> list[TrialGroupSummary]:
    summaries = []

    # Overall by phase.
    for phase in sorted({row.get("phase", "") for row in rows}):
        phase_rows = [row for row in rows if row.get("phase", "") == phase]
        summaries.append(_summarize_trial_group(phase_rows, session_id, participant_id, phase, "all"))

    # By CPI within each phase.
    phases = sorted({row.get("phase", "") for row in rows})
    for phase in phases:
        phase_rows = [row for row in rows if row.get("phase", "") == phase]
        for cpi in sorted({row.get("cpi", "") for row in phase_rows}, key=lambda x: int(x) if str(x).isdigit() else x):
            cpi_rows = [row for row in phase_rows if row.get("cpi", "") == cpi]
            summaries.append(_summarize_trial_group(cpi_rows, session_id, participant_id, phase, cpi))

    # Overall all phases and CPI.
    summaries.append(_summarize_trial_group(rows, session_id, participant_id, "all", "all"))
    for cpi in sorted({row.get("cpi", "") for row in rows}, key=lambda x: int(x) if str(x).isdigit() else x):
        cpi_rows = [row for row in rows if row.get("cpi", "") == cpi]
        summaries.append(_summarize_trial_group(cpi_rows, session_id, participant_id, "all", cpi))

    return summaries


def _summarize_session(session_dir: Path, session: dict[str, Any], rows: list[dict[str, str]]) -> SessionSummary:
    participant = session.get("participant") or {}
    display = session.get("display_calibration") or (session.get("hardware") or {}).get("display_calibration") or {}
    event_counts = _event_counts(session_dir)
    measured = [row for row in rows if row.get("phase") == "measured"]
    measured_success = sum(row.get("outcome") == "success" for row in measured)
    measured_miss = sum(row.get("outcome") == "miss" for row in measured)
    measured_timeout = sum(row.get("outcome") == "timeout" for row in measured)
    invalid = sum(_bool(row.get("valid")) is False for row in rows)
    focus_lost = sum(_bool(row.get("focus_lost")) is True for row in rows)
    fullscreen_false = sum(_bool(row.get("client_fullscreen")) is False for row in rows)
    cpis = sorted({row.get("cpi", "") for row in rows}, key=lambda x: int(x) if str(x).isdigit() else x)

    return SessionSummary(
        session_id=session_dir.name,
        participant_id=_participant_id(session, session_dir),
        input_backend=str(session.get("input_backend", "")),
        dominant_hand=str(participant.get("dominant_hand", "")),
        age=participant.get("age", ""),
        n_trials=len(rows),
        n_practice=sum(row.get("phase") == "practice" for row in rows),
        n_measured=len(measured),
        measured_success_rate=_rate(measured_success, len(measured)),
        measured_miss=measured_miss,
        measured_timeout=measured_timeout,
        invalid=invalid,
        focus_lost_count=focus_lost,
        fullscreen_false_count=fullscreen_false,
        cpi_values=";".join(cpis),
        display_resolution="x".join(str(v) for v in display.get("resolution_px", [])),
        display_size_cm="x".join(str(v) for v in display.get("physical_size_cm", [])),
        browser_estimated_resolution="x".join(str(v) for v in display.get("browser_estimated_resolution_px", [])),
        fixed_profile_warning=str(display.get("fixed_profile_warning", "")),
        session_complete=event_counts.get("session_complete", 0) > 0,
    )


def _summarize_raw_by_cpi(
    session_dir: Path,
    participant_id: str,
) -> list[RawCpiSummary]:
    raw_path = session_dir / "raw_mouse_events.csv"
    if not raw_path.exists():
        return []

    grouped: dict[tuple[str, str], dict[str, list[float] | int]] = {}
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cpi = row.get("cpi", "")
            phase = row.get("phase", "")
            dx = _float(row.get("dx_counts")) or 0.0
            dy = _float(row.get("dy_counts")) or 0.0
            count_mag = math.hypot(dx, dy)
            if count_mag <= 0:
                continue
            key = (phase, cpi)
            bucket = grouped.setdefault(
                key,
                {
                    "n": 0,
                    "count_mag": [],
                    "cursor_delta_mm": [],
                    "raw_delta_mm": [],
                    "dt_ms": [],
                },
            )
            bucket["n"] = int(bucket["n"]) + 1
            cast_count_mag = bucket["count_mag"]
            cast_cursor_delta = bucket["cursor_delta_mm"]
            cast_raw_delta = bucket["raw_delta_mm"]
            cast_dt = bucket["dt_ms"]
            assert isinstance(cast_count_mag, list)
            assert isinstance(cast_cursor_delta, list)
            assert isinstance(cast_raw_delta, list)
            assert isinstance(cast_dt, list)
            cast_count_mag.append(count_mag)
            cursor_dx = _float(row.get("cursor_delta_x_m")) or 0.0
            cursor_dy = _float(row.get("cursor_delta_y_m")) or 0.0
            cast_cursor_delta.append(math.hypot(cursor_dx, cursor_dy) * 1000.0)
            raw_dx = _float(row.get("raw_dx_m")) or 0.0
            raw_dy = _float(row.get("raw_dy_m")) or 0.0
            cast_raw_delta.append(math.hypot(raw_dx, raw_dy) * 1000.0)
            dt_ns = _float(row.get("dt_ns"))
            if dt_ns is not None and dt_ns > 0:
                cast_dt.append(dt_ns / 1e6)

    summaries: list[RawCpiSummary] = []
    for (phase, cpi), bucket in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], int(item[0][1]) if str(item[0][1]).isdigit() else item[0][1]),
    ):
        count_mag = bucket["count_mag"]
        cursor_delta = bucket["cursor_delta_mm"]
        raw_delta = bucket["raw_delta_mm"]
        dt_ms = bucket["dt_ms"]
        assert isinstance(count_mag, list)
        assert isinstance(cursor_delta, list)
        assert isinstance(raw_delta, list)
        assert isinstance(dt_ms, list)
        summaries.append(
            RawCpiSummary(
                session_id=session_dir.name,
                participant_id=participant_id,
                phase=phase,
                cpi=cpi,
                moving_raw_events=int(bucket["n"]),
                median_count_magnitude=_median(count_mag),
                mean_count_magnitude=_mean(count_mag),
                mean_cursor_delta_mm=_mean(cursor_delta),
                median_cursor_delta_mm=_median(cursor_delta),
                mean_raw_delta_mm=_mean(raw_delta),
                median_dt_ms=_median(dt_ms),
                max_dt_ms=max(dt_ms) if dt_ms else None,
            )
        )

    # Also add all-phase CPI summaries.
    all_by_cpi: dict[str, list[RawCpiSummary]] = {}
    for item in summaries:
        all_by_cpi.setdefault(item.cpi, []).append(item)
    for cpi, items in sorted(all_by_cpi.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else item[0]):
        # Use weighted approximations for means and simple medians across phase medians.
        n_total = sum(item.moving_raw_events for item in items)
        if n_total <= 0:
            continue
        summaries.append(
            RawCpiSummary(
                session_id=session_dir.name,
                participant_id=participant_id,
                phase="all",
                cpi=cpi,
                moving_raw_events=n_total,
                median_count_magnitude=_median(item.median_count_magnitude for item in items),
                mean_count_magnitude=sum((item.mean_count_magnitude or 0.0) * item.moving_raw_events for item in items) / n_total,
                mean_cursor_delta_mm=sum((item.mean_cursor_delta_mm or 0.0) * item.moving_raw_events for item in items) / n_total,
                median_cursor_delta_mm=_median(item.median_cursor_delta_mm for item in items),
                mean_raw_delta_mm=sum((item.mean_raw_delta_mm or 0.0) * item.moving_raw_events for item in items) / n_total,
                median_dt_ms=_median(item.median_dt_ms for item in items),
                max_dt_ms=max((item.max_dt_ms or 0.0) for item in items),
            )
        )

    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    session_summaries: list[SessionSummary],
    cpi_summaries: list[TrialGroupSummary],
    raw_summaries: list[RawCpiSummary],
) -> None:
    measured_cpi = [
        item for item in cpi_summaries
        if item.phase == "measured" and item.cpi != "all"
    ]
    raw_measured = [
        item for item in raw_summaries
        if item.phase == "measured"
    ]

    lines = [
        "# Human experiment quality check",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Session overview",
        "",
        _markdown_table(
            [
                "Participant",
                "Session",
                "Trials",
                "Measured",
                "Measured success",
                "Invalid",
                "Complete",
                "Display warning",
            ],
            [
                [
                    s.participant_id,
                    s.session_id,
                    s.n_trials,
                    s.n_measured,
                    "" if s.measured_success_rate is None else f"{s.measured_success_rate:.1%}",
                    s.invalid,
                    "yes" if s.session_complete else "no",
                    s.fixed_profile_warning or "",
                ]
                for s in session_summaries
            ],
        ),
        "",
        "## Measured trials by CPI",
        "",
        _markdown_table(
            [
                "Participant",
                "CPI",
                "n",
                "Success",
                "Miss",
                "Timeout",
                "Success rate",
                "Mean movement time",
                "Mean click error",
            ],
            [
                [
                    item.participant_id,
                    item.cpi,
                    item.n_trials,
                    item.success,
                    item.miss,
                    item.timeout,
                    "" if item.success_rate is None else f"{item.success_rate:.1%}",
                    _fmt(item.mean_movement_time_s, 3),
                    _fmt(item.mean_click_error_m, 4),
                ]
                for item in measured_cpi
            ],
        ),
        "",
        "## Raw movement sanity check by CPI",
        "",
        _markdown_table(
            [
                "Participant",
                "CPI",
                "moving events",
                "median abs counts",
                "mean cursor delta (mm/event)",
                "median dt (ms)",
            ],
            [
                [
                    item.participant_id,
                    item.cpi,
                    item.moving_raw_events,
                    _fmt(item.median_count_magnitude, 2),
                    _fmt(item.mean_cursor_delta_mm, 4),
                    _fmt(item.median_dt_ms, 3),
                ]
                for item in raw_measured
            ],
        ),
        "",
        "## QC notes",
        "",
    ]

    warnings: list[str] = []
    for s in session_summaries:
        if not s.session_complete:
            warnings.append(f"- {s.participant_id}: session_complete event is missing.")
        if s.invalid:
            warnings.append(f"- {s.participant_id}: {s.invalid} invalid trials.")
        if s.measured_timeout:
            warnings.append(f"- {s.participant_id}: {s.measured_timeout} measured timeouts.")
        if s.fixed_profile_warning:
            warnings.append(f"- {s.participant_id}: display warning: {s.fixed_profile_warning}")

    if warnings:
        lines.extend(warnings)
    else:
        lines.append("- No major QC warning detected.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_quality_check(data_root: Path, output_dir: Path) -> tuple[Path, int]:
    sessions = _session_dirs(data_root)
    if not sessions:
        raise SystemExit(f"No sessions with session.json and trials.csv found in {data_root}")

    session_rows: list[dict[str, Any]] = []
    cpi_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    session_summaries: list[SessionSummary] = []
    cpi_summaries: list[TrialGroupSummary] = []
    raw_summaries: list[RawCpiSummary] = []

    for session_dir in sessions:
        session = _read_json(session_dir / "session.json")
        trial_rows = _read_csv_rows(session_dir / "trials.csv")
        participant = _participant_id(session, session_dir)

        session_summary = _summarize_session(session_dir, session, trial_rows)
        session_summaries.append(session_summary)
        session_rows.append(session_summary.to_row())

        grouped_trials = _group_trial_rows(trial_rows, session_dir.name, participant)
        cpi_summaries.extend(grouped_trials)
        cpi_rows.extend(item.to_row() for item in grouped_trials)

        grouped_raw = _summarize_raw_by_cpi(session_dir, participant)
        raw_summaries.extend(grouped_raw)
        raw_rows.extend(item.to_row() for item in grouped_raw)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "session_summary.csv", session_rows)
    _write_csv(output_dir / "trial_cpi_summary.csv", cpi_rows)
    _write_csv(output_dir / "raw_cpi_summary.csv", raw_rows)
    _write_report(output_dir / "quality_report.md", session_summaries, cpi_summaries, raw_summaries)
    return output_dir, len(sessions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing human experiment session folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where quality-check outputs should be written.",
    )
    args = parser.parse_args()

    data_root = args.root.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = data_root / "qc_reports" / stamp
    else:
        output_dir = output_dir.resolve()

    out, n_sessions = run_quality_check(data_root, output_dir)
    print(f"Quality check complete for {n_sessions} sessions.")
    print(f"Output directory: {out}")
    print(f"- {out / 'quality_report.md'}")
    print(f"- {out / 'session_summary.csv'}")
    print(f"- {out / 'trial_cpi_summary.csv'}")
    print(f"- {out / 'raw_cpi_summary.csv'}")


if __name__ == "__main__":
    main()
