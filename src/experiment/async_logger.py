"""Process-backed logger so CSV serialization never runs in the render process."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
from pathlib import Path
import queue
from typing import Any

from ..utils.trajectory_features import CURVATURE_FEATURE_NAMES


RAW_FIELDS = [
    "session_id", "trial_id", "phase", "trial_stage", "block_index", "cpi",
    "mousepad_condition", "input_backend", "event_index",
    "timestamp_ns", "time_from_onset_ns", "device_handle", "device_name",
    "os_cursor_x_px", "os_cursor_y_px", "browser_timestamp_ms", "raw_flags",
    "button_flags", "button_data", "dx_counts", "dy_counts", "dt_ns",
    "raw_dx_m", "raw_dy_m", "raw_trial_x_m", "raw_trial_y_m",
    "raw_session_x_m", "raw_session_y_m", "raw_path_length_m",
    "raw_displacement_from_trial_start_m", "raw_speed_m_s",
    "raw_angle_rad", "raw_acceleration_x_m_s2", "raw_acceleration_y_m_s2",
    "raw_acceleration_m_s2",
    "cursor_delta_x_m", "cursor_delta_y_m", "cursor_heading_rad",
    "cursor_path_length_m",
    "cursor_x_m", "cursor_y_m", "velocity_x_m_s", "velocity_y_m_s",
    "speed_m_s", "cursor_acceleration_x_m_s2", "cursor_acceleration_y_m_s2",
    "cursor_acceleration_m_s2", "target_x_m", "target_y_m", "target_radius_m",
    "distance_to_target_m", "angle_to_target_rad", "heading_error_rad",
    "inside_target",
]

FRAME_FIELDS = [
    "session_id", "trial_id", "phase", "trial_stage", "block_index", "cpi",
    "mousepad_condition", "input_backend", "frame_index",
    "frame_start_ns", "present_return_ns", "time_from_onset_ns",
    "frame_interval_ns", "late_by_ns", "raw_events_consumed", "cursor_x_m",
    "cursor_y_m", "target_x_m", "target_y_m", "target_radius_m",
    "distance_to_target_m", "inside_target", "window_focused",
    "browser_timestamp_ms", "browser_interval_ms", "browser_time_origin_ms",
    "document_visible", "pointer_locked", "raw_event_sequence",
]

TRIAL_FIELDS = [
    "session_id", "trial_id", "template_id", "phase", "trial_stage", "block_index", "cpi",
    "mousepad_condition", "input_backend",
    "attempt", "valid", "invalid_reason", "outcome", "success", "timeout",
    "onset_ns", "first_movement_ns", "click_ns", "end_ns", "reaction_time_s",
    "movement_time_s", "total_time_s", "cursor_start_x_m", "cursor_start_y_m",
    "cursor_end_x_m", "cursor_end_y_m", "cursor_displacement_m",
    "target_x_m", "target_y_m", "target_radius_m", "target_width_m",
    "rendered_target_radius_px", "rendered_target_radius_m", "rendered_fitts_id",
    "target_distance_m", "fitts_id", "click_x_m", "click_y_m",
    "click_raw_trial_x_m", "click_raw_trial_y_m",
    "click_raw_session_x_m", "click_raw_session_y_m",
    "click_error_m", "minimum_target_distance_m", "path_length_m",
    "straight_line_efficiency", "raw_event_count", "frame_count",
    "late_frame_count", "max_frame_interval_ns", "focus_lost",
    "raw_start_x_m", "raw_start_y_m", "raw_end_x_m", "raw_end_y_m",
    "raw_session_start_x_m", "raw_session_start_y_m",
    "raw_session_end_x_m", "raw_session_end_y_m",
    "raw_path_length_m", "raw_displacement_m",
    "raw_straight_line_efficiency", "raw_mean_speed_m_s",
    "raw_max_speed_m_s", "raw_event_rate_hz",
    "mean_polling_interval_ns", "std_polling_interval_ns",
    "max_polling_gap_ns",
    *CURVATURE_FEATURE_NAMES,
    "browser_onset_ms", "browser_first_movement_ms", "browser_click_ms",
    "browser_end_ms", "pointer_lock_raw_requested", "pointer_lock_raw_granted",
    "client_fullscreen", "client_width_px", "client_height_px", "device_pixel_ratio",
    "client_inner_width_css_px", "client_inner_height_css_px",
    "client_outer_width_css_px", "client_outer_height_css_px",
    "screen_width_css_px", "screen_height_css_px",
    "screen_avail_width_css_px", "screen_avail_height_css_px",
    "visual_viewport_width_css_px", "visual_viewport_height_css_px",
    "canvas_css_width_px", "canvas_css_height_px",
    "canvas_bounding_left_px", "canvas_bounding_top_px",
    "canvas_bounding_width_px", "canvas_bounding_height_px",
    "reported_screen_width_cm", "reported_screen_height_cm",
    "reported_resolution_width_px", "reported_resolution_height_px",
    "reported_refresh_hz", "display_diagonal_in", "fixed_profile_warning",
    "physical_px_per_m_x",
    "physical_px_per_m_y", "render_scale_px_per_virtual_m",
    "rendered_task_width_px", "rendered_task_height_px",
    "rendered_task_width_physical_m", "rendered_task_height_physical_m",
    "rendered_target_radius_physical_m",
]


def _open_csv(path: Path, fields: list[str]):
    handle = path.open("w", newline="", encoding="utf-8", buffering=1024 * 1024)
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def _writer_main(output_dir: str, work_queue: mp.Queue) -> None:
    root = Path(output_dir)
    raw_file, raw_writer = _open_csv(root / "raw_mouse_events.csv", RAW_FIELDS)
    frame_file, frame_writer = _open_csv(root / "frames.csv", FRAME_FIELDS)
    trial_file, trial_writer = _open_csv(root / "trials.csv", TRIAL_FIELDS)
    event_file = (root / "experiment_events.jsonl").open(
        "w", encoding="utf-8", buffering=1024 * 1024
    )
    try:
        while True:
            message = work_queue.get()
            if message is None:
                break
            kind = message[0]
            if kind == "trial":
                _, trial, raw_rows, frame_rows, events = message
                raw_writer.writerows(raw_rows)
                frame_writer.writerows(frame_rows)
                trial_writer.writerow(trial)
                for event in events:
                    event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                raw_file.flush()
                frame_file.flush()
                trial_file.flush()
                event_file.flush()
            elif kind == "event":
                event_file.write(json.dumps(message[1], ensure_ascii=False) + "\n")
                event_file.flush()
    finally:
        raw_file.close()
        frame_file.close()
        trial_file.close()
        event_file.close()


class AsyncExperimentLogger:
    def __init__(self, output_dir: Path, session_metadata: dict[str, Any], queue_size: int = 12):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / "session.json").write_text(
            json.dumps(session_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=queue_size)
        self._process = context.Process(
            target=_writer_main,
            args=(str(output_dir), self._queue),
            name="pnc-experiment-writer",
        )
        self._process.start()
        self.queue_backpressure_count = 0

    def submit_trial(self, trial: dict, raw_rows: list[dict], frame_rows: list[dict], events: list[dict]):
        message = ("trial", trial, raw_rows, frame_rows, events)
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            # This happens between trials, never inside the 240 Hz render loop.
            self.queue_backpressure_count += 1
            self._queue.put(message)

    def submit_event(self, event: dict):
        self._queue.put(("event", event))

    def close(self):
        if self._process is None:
            return
        self._queue.put(None)
        self._process.join(timeout=30)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            raise RuntimeError("The logging process did not stop cleanly")
        if self._process.exitcode != 0:
            raise RuntimeError(f"The logging process exited with code {self._process.exitcode}")
        self._queue.close()
        self._process = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
