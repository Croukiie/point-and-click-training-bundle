"""Local web interface for the human point-and-click experiment.

The HTTP server only binds to localhost. On Windows it can use native Raw Input.
On macOS/Linux it falls back to browser Pointer Lock movement for pilot/testing.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import math
from pathlib import Path
import platform
import re
import sys
import threading
import time
import webbrowser
from typing import Any

from aiohttp import WSMsgType, web
import numpy as np
import yaml

from .async_logger import AsyncExperimentLogger
from .task_design import counterbalanced_cpi_order, make_trial_templates

RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
try:
    if sys.platform != "win32":
        raise ImportError("Windows Raw Input is unavailable on this platform")
    from .raw_input_win32 import RawInputMouse
except (ImportError, AttributeError):
    RawInputMouse = None


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).with_name("web")
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
INCH_M = 0.0254
BACKEND_WINDOWS_RAW_INPUT = "windows_raw_input"
BACKEND_BROWSER_POINTER_LOCK = "browser_pointer_lock"


def resolve_input_backend(requested: str) -> str:
    if requested == "auto":
        return (
            BACKEND_WINDOWS_RAW_INPUT
            if sys.platform == "win32" and RawInputMouse is not None
            else BACKEND_BROWSER_POINTER_LOCK
        )
    if requested == BACKEND_WINDOWS_RAW_INPUT and RawInputMouse is None:
        raise RuntimeError("Windows Raw Input backend is only available on Windows")
    return requested


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_participant_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())[:64]
    if not cleaned:
        raise ValueError("Participant ID is required")
    return cleaned


def _number_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two numeric values")
    first, second = float(value[0]), float(value[1])
    if first <= 0 or second <= 0:
        raise ValueError(f"{label} values must be positive")
    return first, second


def normalize_display_calibration(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Display calibration is required")
    width_cm, height_cm = _number_pair(value.get("physical_size_cm"), "Screen size in cm")
    width_m, height_m = width_cm / 100.0, height_cm / 100.0
    resolution_w, resolution_h = _number_pair(value.get("resolution_px"), "Screen resolution")
    resolution = [int(round(resolution_w)), int(round(resolution_h))]
    if resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError("Screen resolution values must be positive")
    refresh_hz = float(value.get("refresh_hz"))
    if refresh_hz <= 0:
        raise ValueError("Refresh rate must be positive")
    px_per_m = [resolution[0] / width_m, resolution[1] / height_m]
    return {
        "source": str(value.get("source", "fixed_protocol_profile")),
        "physical_size_cm": [width_cm, height_cm],
        "physical_size_m": [width_m, height_m],
        "resolution_px": resolution,
        "refresh_hz": refresh_hz,
        "px_per_m": px_per_m,
        "diagonal_in": math.hypot(width_cm, height_cm) / 2.54,
        "browser_screen_css_px": value.get("browser_screen_css_px"),
        "browser_screen_avail_css_px": value.get("browser_screen_avail_css_px"),
        "browser_viewport_css_px": value.get("browser_viewport_css_px"),
        "browser_outer_window_css_px": value.get("browser_outer_window_css_px"),
        "browser_visual_viewport_css_px": value.get("browser_visual_viewport_css_px"),
        "browser_device_pixel_ratio": value.get("browser_device_pixel_ratio"),
        "browser_estimated_resolution_px": value.get("browser_estimated_resolution_px"),
        "fixed_profile_warning": str(value.get("fixed_profile_warning", "")),
        "notes": str(value.get("notes", "")),
    }


def validate_fixed_display_profile(
    calibration: dict[str, Any],
    display_config: dict[str, Any],
) -> None:
    if not bool(display_config.get("enforce_fixed_profile", False)):
        return
    expected_resolution = [int(v) for v in display_config["resolution_px"]]
    if calibration["resolution_px"] != expected_resolution:
        raise ValueError(
            "This protocol requires a Full HD display: "
            f"{expected_resolution[0]}x{expected_resolution[1]} px"
        )
    browser_resolution = calibration.get("browser_estimated_resolution_px")
    if isinstance(browser_resolution, (list, tuple)) and len(browser_resolution) == 2:
        estimated_resolution = [int(round(float(v))) for v in browser_resolution]
        if any(abs(actual - expected) > 2 for actual, expected in zip(estimated_resolution, expected_resolution)):
            raise ValueError(
                "The browser does not appear to be running on the fixed Full HD display. "
                f"Expected {expected_resolution[0]}x{expected_resolution[1]} px, "
                f"browser reported approximately {estimated_resolution[0]}x{estimated_resolution[1]} px."
            )
    expected_size_cm = [float(v) * 100.0 for v in display_config["physical_size_m"]]
    tolerance_cm = float(display_config.get("physical_tolerance_cm", 1.5))
    reported_size_cm = calibration["physical_size_cm"]
    if any(
        abs(float(reported) - expected) > tolerance_cm
        for reported, expected in zip(reported_size_cm, expected_size_cm)
    ):
        raise ValueError(
            "This protocol requires a 24 inch display profile: "
            f"approximately {expected_size_cm[0]:.1f}x{expected_size_cm[1]:.1f} cm"
        )


class WebExperimentServer:
    def __init__(self, config: dict[str, Any], output_root: Path, input_backend: str = "auto"):
        self.config = config
        self.exp = config["experiment"]
        self.output_root = output_root
        self.input_backend = resolve_input_backend(input_backend)
        self.raw_mouse = (
            RawInputMouse()
            if self.input_backend == BACKEND_WINDOWS_RAW_INPUT
            else None
        )
        self.ws: web.WebSocketResponse | None = None
        self.raw_task: asyncio.Task | None = None
        self.timeout_task: asyncio.Task | None = None
        self.logger: AsyncExperimentLogger | None = None
        self.session_id: str | None = None
        self.participant_id: str | None = None
        self.cpi_order: list[int] = []
        self.templates = []
        self.template_by_id = {}
        self.trial: dict[str, Any] | None = None
        self.trial_sequence = 0
        self.display_calibration: dict[str, Any] | None = None

    async def start(self, app: web.Application):
        if self.raw_mouse is not None:
            self.raw_mouse.start()
            self.raw_task = asyncio.create_task(self._raw_pump(), name="raw-input-web-pump")

    async def stop(self, app: web.Application):
        if self.timeout_task:
            self.timeout_task.cancel()
            try:
                await self.timeout_task
            except asyncio.CancelledError:
                pass
        if self.raw_task:
            self.raw_task.cancel()
            try:
                await self.raw_task
            except asyncio.CancelledError:
                pass
        if self.raw_mouse is not None:
            self.raw_mouse.stop()
        if self.logger:
            self.logger.submit_event({
                "event": "server_stop", "session_id": self.session_id,
                "timestamp_ns": time.perf_counter_ns(),
            })
            logger, self.logger = self.logger, None
            await asyncio.to_thread(logger.close)

    async def _close_session_for_new_participant(
        self,
        event_name: str,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        session_id = self.session_id
        if self.logger:
            event = {
                "event": event_name,
                "session_id": self.session_id,
                "timestamp_ns": time.perf_counter_ns(),
            }
            if extra:
                event.update(extra)
            self.logger.submit_event(event)
            logger, self.logger = self.logger, None
            await asyncio.to_thread(logger.close)
        self.session_id = None
        self.participant_id = None
        self.cpi_order = []
        self.templates = []
        self.template_by_id = {}
        self.trial = None
        self.trial_sequence = 0
        self.display_calibration = None
        if self.timeout_task:
            self.timeout_task.cancel()
            self.timeout_task = None
        return session_id

    async def index(self, request: web.Request):
        return web.FileResponse(STATIC_DIR / "index.html")

    async def health(self, request: web.Request):
        return web.json_response({
            "status": "ok",
            "input_backend": self.input_backend,
            "raw_input": self.input_backend == BACKEND_WINDOWS_RAW_INPUT,
            "session_active": self.logger is not None,
        })

    async def websocket(self, request: web.Request):
        if self.ws is not None and not self.ws.closed:
            return web.Response(status=409, text="An experiment browser is already connected")
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        self.ws = ws
        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    try:
                        await self._handle_message(message.json())
                    except (ValueError, KeyError, TypeError) as exc:
                        await self._send({"type": "error", "message": str(exc)})
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            if self.trial and self.trial.get("state") == "active":
                await self._finish_trial("connection_lost", valid=False)
            self.ws = None
        return ws

    async def _send(self, payload: dict[str, Any]):
        if self.ws is not None and not self.ws.closed:
            await self.ws.send_json(payload)

    async def _handle_message(self, data: dict[str, Any]):
        message_type = data.get("type")
        if message_type == "hello":
            await self._start_session(data)
        elif message_type == "start_trial":
            await self._prepare_trial(data)
        elif message_type == "target_onset":
            await self._target_onset(data)
        elif message_type == "client_input_batch":
            await self._client_input_batch(data)
        elif message_type == "invalidate_trial":
            if self.trial and self.trial.get("state") == "active":
                reason = str(data.get("reason", "client_invalidated"))
                await self._finish_trial(reason, valid=False)
        elif message_type == "trial_client_data":
            await self._save_trial(data)
        elif message_type == "pause_event":
            if self.logger:
                self.logger.submit_event({
                    "event": str(data.get("event", "pause_event")),
                    "session_id": self.session_id,
                    "trial_id": data.get("trial_id"),
                    "phase": data.get("phase"),
                    "block_index": data.get("block_index"),
                    "phase_index": data.get("phase_index"),
                    "browser_ms": data.get("browser_ms"),
                    "timestamp_ns": time.perf_counter_ns(),
                })
        elif message_type == "session_complete":
            completed_session_id = await self._close_session_for_new_participant(
                "session_complete"
            )
            await self._send({
                "type": "session_saved", "session_id": completed_session_id,
                "aborted": False, "ready_for_new_session": True,
            })
        elif message_type == "abort_session":
            if self.trial and self.trial.get("state") == "active":
                await self._finish_trial("session_aborted", valid=False)
                return
            aborted_session_id = await self._close_session_for_new_participant(
                "session_aborted",
                {
                    "reason": data.get("reason", "participant_menu_stop"),
                    "browser_ms": data.get("browser_ms"),
                },
            )
            await self._send({
                "type": "session_saved", "session_id": aborted_session_id,
                "aborted": True, "ready_for_new_session": True,
            })
        elif message_type == "ping":
            await self._send({
                "type": "pong", "client_ms": data.get("client_ms"),
                "server_perf_ns": time.perf_counter_ns(), "server_epoch_ns": time.time_ns(),
            })
        else:
            raise ValueError(f"Unknown message type: {message_type}")

    async def _start_session(self, data: dict[str, Any]):
        if self.logger is not None:
            raise ValueError("A session is already active; restart the server for a new participant")
        participant = data.get("participant", {})
        participant_id = safe_participant_id(str(participant.get("id", "")))
        hardware = dict(data.get("hardware", {}))
        self.display_calibration = normalize_display_calibration(
            hardware.get("display_calibration")
        )
        validate_fixed_display_profile(self.display_calibration, self.config["display"])
        hardware["display_calibration"] = self.display_calibration
        hardware["declared_resolution_px"] = self.display_calibration["resolution_px"]
        hardware["declared_refresh_hz"] = self.display_calibration["refresh_hz"]
        self.participant_id = participant_id
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{participant_id}_web"
        cpi_values = [int(v) for v in self.exp["cpi_values"]]
        self.cpi_order = counterbalanced_cpi_order(
            participant_id, cpi_values, int(self.exp["seed"])
        )
        count = int(self.exp["practice_trials_per_cpi"]) + int(
            self.exp["measured_trials_per_cpi"]
        )
        virtual_size = tuple(float(v) for v in self.config["display"]["virtual_size_m"])
        self.templates = make_trial_templates(
            count, virtual_size, tuple(float(v) for v in self.exp["target_id_range"]),
            int(self.exp["seed"]),
        )
        self.template_by_id = {item.template_id: item for item in self.templates}
        output_dir = self.output_root / self.session_id
        clock = time.get_clock_info("perf_counter")
        metadata = {
            "schema_version": 1,
            "interface": "local_html_with_python_recorder",
            "input_backend": self.input_backend,
            "session_id": self.session_id,
            "participant": participant,
            "hardware": hardware,
            "display_calibration": self.display_calibration,
            "browser": data.get("browser", {}),
            "task": {
                "config": self.config,
                "cpi_order": self.cpi_order,
                "target_shape": "circle",
                "target_motion": "fixed",
                "first_click_ends_trial": True,
                "cursor_source": (
                    "Windows Raw Input reconstructed by Python"
                    if self.input_backend == BACKEND_WINDOWS_RAW_INPUT
                    else "Browser Pointer Lock movement reconstructed by the local web client"
                ),
            },
            "timing": {
                "server_perf_counter_resolution_s": clock.resolution,
                "target_onset": "server timestamp upon receiving the post-render browser message",
                "raw_events": (
                    "server perf_counter_ns when WM_INPUT is handled"
                    if self.input_backend == BACKEND_WINDOWS_RAW_INPUT
                    else "browser performance.now timestamps aligned to server target onset"
                ),
                "frames": "requestAnimationFrame timestamps from the browser",
            },
            "software": {
                "python": sys.version, "platform": platform.platform(),
            },
            "created_local": datetime.now().astimezone().isoformat(),
        }
        self.logger = AsyncExperimentLogger(
            output_dir, metadata, int(self.config["logging"]["writer_queue_size"])
        )
        self.logger.submit_event({
            "event": "session_start", "session_id": self.session_id,
            "timestamp_ns": time.perf_counter_ns(),
            "input_backend": self.input_backend,
        })
        await self._send({
            "type": "session_ready", "session_id": self.session_id,
            "cpi_order": self.cpi_order,
            "templates": [item.to_dict() for item in self.templates],
            "experiment": self.exp,
            "display": self.config["display"],
            "display_calibration": self.display_calibration,
            "appearance": self.config["appearance"],
            "input_backend": self.input_backend,
        })

    async def _prepare_trial(self, data: dict[str, Any]):
        if self.logger is None:
            raise ValueError("Start a session first")
        if self.trial is not None:
            raise ValueError("The previous trial has not been saved yet")
        template_id = int(data["template_id"])
        if template_id not in self.template_by_id:
            raise ValueError("Unknown trial template")
        cpi = int(data["cpi"])
        if cpi not in self.cpi_order:
            raise ValueError("Invalid CPI condition")
        phase = str(data["phase"])
        if phase not in {"practice", "measured"}:
            raise ValueError("Invalid trial phase")
        template = self.template_by_id[template_id]
        self.trial_sequence += 1
        cursor = np.array([template.cursor_x_m, template.cursor_y_m], dtype=np.float64)
        target = np.array([template.target_x_m, template.target_y_m], dtype=np.float64)
        if self.raw_mouse is not None:
            self.raw_mouse.drain()
        self.trial = {
            "state": "ready", "trial_id": self.trial_sequence,
            "template": template, "phase": phase, "cpi": cpi,
            "block_index": int(data["block_index"]), "attempt": int(data.get("attempt", 1)),
            "cursor": cursor, "cursor_start": cursor.copy(), "target": target,
            "path_length": 0.0, "minimum_distance": float(np.linalg.norm(cursor - target)),
            "first_movement_ns": None, "previous_raw_ns": None,
            "raw_rows": [], "events": [], "event_index": 0,
            "click_ns": None, "click_pos": None, "onset_ns": None,
            "browser_onset_ms": None,
            "input_backend": self.input_backend,
        }
        await self._send({
            "type": "trial_ready", "trial_id": self.trial_sequence,
            "template": template.to_dict(), "cursor_x_m": float(cursor[0]),
            "cursor_y_m": float(cursor[1]),
        })

    async def _target_onset(self, data: dict[str, Any]):
        if not self.trial or self.trial.get("state") != "ready":
            raise ValueError("No trial is ready for onset")
        if self.raw_mouse is not None:
            self.raw_mouse.drain()
        onset_ns = time.perf_counter_ns()
        self.trial["onset_ns"] = onset_ns
        self.trial["browser_onset_ms"] = float(data["browser_ms"])
        self.trial["state"] = "active"
        self.trial["events"].append({
            "event": "target_onset", "session_id": self.session_id,
            "trial_id": self.trial["trial_id"], "timestamp_ns": onset_ns,
            "browser_ms": self.trial["browser_onset_ms"],
        })
        await self._send({
            "type": "trial_active", "trial_id": self.trial["trial_id"],
            "server_onset_ns": onset_ns,
        })
        if self.input_backend == BACKEND_BROWSER_POINTER_LOCK:
            if self.timeout_task:
                self.timeout_task.cancel()
            self.timeout_task = asyncio.create_task(
                self._timeout_watch(self.trial["trial_id"], onset_ns),
                name="browser-pointer-lock-timeout",
            )

    async def _timeout_watch(self, trial_id: int, onset_ns: int):
        await asyncio.sleep(float(self.exp["timeout_s"]))
        trial = self.trial
        if (
            trial
            and trial.get("state") == "active"
            and int(trial.get("trial_id")) == int(trial_id)
            and int(trial.get("onset_ns")) == int(onset_ns)
        ):
            await self._finish_trial("timeout", valid=True)

    def _browser_event_timestamp_ns(self, trial: dict[str, Any], event: dict[str, Any]) -> int:
        browser_ms = event.get("browser_ms")
        if browser_ms is None or trial.get("browser_onset_ms") is None:
            return time.perf_counter_ns()
        return int(
            trial["onset_ns"]
            + round((float(browser_ms) - float(trial["browser_onset_ms"])) * 1e6)
        )

    def _append_pointer_event_row(
        self,
        trial: dict[str, Any],
        *,
        timestamp_ns: int,
        dx_counts: int,
        dy_counts: int,
        button_flags: int,
        button_data: int = 0,
        browser_ms: float | None = None,
        cursor_x_m: float | None = None,
        cursor_y_m: float | None = None,
        device_handle: int = 0,
        device_name: str = "",
        os_cursor_x_px: int | None = None,
        os_cursor_y_px: int | None = None,
        raw_flags: int = 0,
    ) -> tuple[dict[str, Any], bool]:
        old_cursor = trial["cursor"].copy()
        virtual_width, virtual_height = (
            float(v) for v in self.config["display"]["virtual_size_m"]
        )

        if cursor_x_m is not None and cursor_y_m is not None:
            trial["cursor"][0] = float(cursor_x_m)
            trial["cursor"][1] = float(cursor_y_m)
        else:
            scale = INCH_M / float(trial["cpi"]) * float(self.exp["mouse_gain"])
            trial["cursor"][0] += dx_counts * scale
            trial["cursor"][1] -= dy_counts * scale

        trial["cursor"][0] = np.clip(trial["cursor"][0], -virtual_width / 2, virtual_width / 2)
        trial["cursor"][1] = np.clip(trial["cursor"][1], -virtual_height / 2, virtual_height / 2)

        movement = trial["cursor"] - old_cursor
        movement_norm = float(np.linalg.norm(movement))
        trial["path_length"] += movement_norm
        if movement_norm > 0 and trial["first_movement_ns"] is None:
            trial["first_movement_ns"] = timestamp_ns
        previous_ns = trial["previous_raw_ns"]
        dt_ns = 0 if previous_ns is None else max(0, timestamp_ns - previous_ns)
        velocity = movement / (dt_ns / 1e9) if dt_ns else np.zeros(2)
        distance = float(np.linalg.norm(trial["cursor"] - trial["target"]))
        trial["minimum_distance"] = min(trial["minimum_distance"], distance)
        inside = distance <= trial["template"].target_radius_m
        trial["event_index"] += 1

        row = {
            "session_id": self.session_id, "trial_id": trial["trial_id"],
            "phase": trial["phase"], "block_index": trial["block_index"],
            "cpi": trial["cpi"], "input_backend": self.input_backend,
            "event_index": trial["event_index"],
            "timestamp_ns": timestamp_ns,
            "time_from_onset_ns": timestamp_ns - trial["onset_ns"],
            "device_handle": device_handle, "device_name": device_name,
            "os_cursor_x_px": os_cursor_x_px,
            "os_cursor_y_px": os_cursor_y_px,
            "browser_timestamp_ms": browser_ms,
            "raw_flags": raw_flags,
            "button_flags": button_flags, "button_data": button_data,
            "dx_counts": dx_counts, "dy_counts": dy_counts, "dt_ns": dt_ns,
            "cursor_x_m": float(trial["cursor"][0]),
            "cursor_y_m": float(trial["cursor"][1]),
            "velocity_x_m_s": float(velocity[0]),
            "velocity_y_m_s": float(velocity[1]),
            "speed_m_s": float(np.linalg.norm(velocity)),
            "target_x_m": trial["template"].target_x_m,
            "target_y_m": trial["template"].target_y_m,
            "target_radius_m": trial["template"].target_radius_m,
            "distance_to_target_m": distance, "inside_target": inside,
        }
        trial["raw_rows"].append(row)
        trial["previous_raw_ns"] = timestamp_ns
        return row, inside

    async def _client_input_batch(self, data: dict[str, Any]):
        if self.input_backend != BACKEND_BROWSER_POINTER_LOCK:
            return
        trial = self.trial
        if not trial or trial.get("state") != "active":
            return
        if int(data.get("trial_id", -1)) != int(trial["trial_id"]):
            return

        finish_outcome = None
        for event in data.get("events", []):
            timestamp_ns = self._browser_event_timestamp_ns(trial, event)
            row, inside = self._append_pointer_event_row(
                trial,
                timestamp_ns=timestamp_ns,
                dx_counts=int(round(float(event.get("dx_counts", 0)))),
                dy_counts=int(round(float(event.get("dy_counts", 0)))),
                button_flags=int(event.get("button_flags", 0)),
                button_data=int(event.get("button_data", 0)),
                browser_ms=event.get("browser_ms"),
                cursor_x_m=event.get("cursor_x_m"),
                cursor_y_m=event.get("cursor_y_m"),
                device_name=BACKEND_BROWSER_POINTER_LOCK,
            )
            if row["button_flags"] & RI_MOUSE_LEFT_BUTTON_DOWN:
                trial["click_ns"] = timestamp_ns
                trial["click_pos"] = trial["cursor"].copy()
                finish_outcome = "success" if inside else "miss"
                trial["events"].append({
                    "event": "click", "session_id": self.session_id,
                    "trial_id": trial["trial_id"], "timestamp_ns": timestamp_ns,
                    "browser_ms": event.get("browser_ms"),
                    "x_m": row["cursor_x_m"], "y_m": row["cursor_y_m"],
                    "success": inside,
                    "input_backend": self.input_backend,
                })
                break

        if finish_outcome:
            await self._finish_trial(finish_outcome, valid=True)

    async def _raw_pump(self):
        if self.raw_mouse is None:
            return
        while True:
            await asyncio.sleep(0.001)
            raw_events = self.raw_mouse.drain()
            trial = self.trial
            if not trial or trial.get("state") != "active":
                continue
            onset_ns = trial["onset_ns"]
            cpi = trial["cpi"]
            scale = INCH_M / float(cpi) * float(self.exp["mouse_gain"])
            virtual_width, virtual_height = (
                float(v) for v in self.config["display"]["virtual_size_m"]
            )
            batch = []
            finish_outcome = None
            for raw in raw_events:
                old_cursor = trial["cursor"].copy()
                trial["cursor"][0] += raw.dx * scale
                trial["cursor"][1] -= raw.dy * scale
                trial["cursor"][0] = np.clip(trial["cursor"][0], -virtual_width / 2, virtual_width / 2)
                trial["cursor"][1] = np.clip(trial["cursor"][1], -virtual_height / 2, virtual_height / 2)
                movement = trial["cursor"] - old_cursor
                movement_norm = float(np.linalg.norm(movement))
                trial["path_length"] += movement_norm
                if movement_norm > 0 and trial["first_movement_ns"] is None:
                    trial["first_movement_ns"] = raw.timestamp_ns
                previous_ns = trial["previous_raw_ns"]
                dt_ns = 0 if previous_ns is None else max(0, raw.timestamp_ns - previous_ns)
                velocity = movement / (dt_ns / 1e9) if dt_ns else np.zeros(2)
                distance = float(np.linalg.norm(trial["cursor"] - trial["target"]))
                trial["minimum_distance"] = min(trial["minimum_distance"], distance)
                inside = distance <= trial["template"].target_radius_m
                trial["event_index"] += 1
                row = {
                    "session_id": self.session_id, "trial_id": trial["trial_id"],
                    "phase": trial["phase"], "block_index": trial["block_index"],
                    "cpi": cpi, "input_backend": self.input_backend,
                    "event_index": trial["event_index"],
                    "timestamp_ns": raw.timestamp_ns,
                    "time_from_onset_ns": raw.timestamp_ns - onset_ns,
                    "device_handle": raw.device_handle, "device_name": raw.device_name,
                    "os_cursor_x_px": raw.os_cursor_x_px,
                    "os_cursor_y_px": raw.os_cursor_y_px,
                    "browser_timestamp_ms": None,
                    "raw_flags": raw.raw_flags,
                    "button_flags": raw.button_flags, "button_data": raw.button_data,
                    "dx_counts": raw.dx, "dy_counts": raw.dy, "dt_ns": dt_ns,
                    "cursor_x_m": float(trial["cursor"][0]),
                    "cursor_y_m": float(trial["cursor"][1]),
                    "velocity_x_m_s": float(velocity[0]),
                    "velocity_y_m_s": float(velocity[1]),
                    "speed_m_s": float(np.linalg.norm(velocity)),
                    "target_x_m": trial["template"].target_x_m,
                    "target_y_m": trial["template"].target_y_m,
                    "target_radius_m": trial["template"].target_radius_m,
                    "distance_to_target_m": distance, "inside_target": inside,
                }
                trial["raw_rows"].append(row)
                batch.append({
                    "sequence": trial["event_index"], "timestamp_ns": raw.timestamp_ns,
                    "cursor_x_m": row["cursor_x_m"], "cursor_y_m": row["cursor_y_m"],
                    "button_flags": raw.button_flags, "inside_target": inside,
                })
                trial["previous_raw_ns"] = raw.timestamp_ns
                if raw.button_flags & RI_MOUSE_LEFT_BUTTON_DOWN:
                    trial["click_ns"] = raw.timestamp_ns
                    trial["click_pos"] = trial["cursor"].copy()
                    finish_outcome = "success" if inside else "miss"
                    trial["events"].append({
                        "event": "click", "session_id": self.session_id,
                        "trial_id": trial["trial_id"], "timestamp_ns": raw.timestamp_ns,
                        "x_m": row["cursor_x_m"], "y_m": row["cursor_y_m"],
                        "success": inside,
                        "input_backend": self.input_backend,
                    })
                    break
            if batch:
                await self._send({
                    "type": "raw_batch", "trial_id": trial["trial_id"], "events": batch,
                })
            if finish_outcome:
                await self._finish_trial(finish_outcome, valid=True)
            elif time.perf_counter_ns() - onset_ns >= int(float(self.exp["timeout_s"]) * 1e9):
                await self._finish_trial("timeout", valid=True)

    async def _finish_trial(self, outcome: str, valid: bool):
        trial = self.trial
        if not trial or trial.get("state") != "active":
            return
        if self.timeout_task and self.timeout_task is not asyncio.current_task():
            self.timeout_task.cancel()
            self.timeout_task = None
        end_ns = trial["click_ns"] or time.perf_counter_ns()
        trial["end_ns"] = end_ns
        trial["outcome"] = outcome
        trial["valid"] = valid
        trial["state"] = "awaiting_client_data"
        trial["events"].append({
            "event": "trial_end", "session_id": self.session_id,
            "trial_id": trial["trial_id"], "timestamp_ns": end_ns,
            "outcome": outcome, "valid": valid,
            "input_backend": self.input_backend,
        })
        await self._send({
            "type": "trial_result", "trial_id": trial["trial_id"],
            "outcome": outcome, "valid": valid, "end_server_ns": end_ns,
            "cursor_x_m": float(trial["cursor"][0]),
            "cursor_y_m": float(trial["cursor"][1]),
        })

    async def _save_trial(self, data: dict[str, Any]):
        trial = self.trial
        if not trial or trial.get("state") != "awaiting_client_data":
            raise ValueError("No completed trial is waiting for browser data")
        if int(data["trial_id"]) != trial["trial_id"]:
            raise ValueError("Trial ID mismatch")
        client = data.get("client", {})
        browser_frames = data.get("frames", [])
        frame_rows = []
        late_count = 0
        max_interval_ns = 0
        nominal_ns = round(1e9 / int(self.exp["refresh_hz"]))
        for index, frame in enumerate(browser_frames, start=1):
            interval_ms = float(frame.get("interval_ms", 0.0))
            interval_ns = round(interval_ms * 1e6)
            late_by_ns = max(0, interval_ns - nominal_ns)
            if interval_ns > nominal_ns * 1.5:
                late_count += 1
            max_interval_ns = max(max_interval_ns, interval_ns)
            distance = math.hypot(
                float(frame["cursor_x_m"]) - trial["template"].target_x_m,
                float(frame["cursor_y_m"]) - trial["template"].target_y_m,
            )
            browser_ms = float(frame["browser_ms"])
            frame_rows.append({
                "session_id": self.session_id, "trial_id": trial["trial_id"],
                "phase": trial["phase"], "block_index": trial["block_index"],
                "cpi": trial["cpi"], "input_backend": self.input_backend,
                "frame_index": index,
                "frame_start_ns": "", "present_return_ns": "",
                "time_from_onset_ns": round(
                    (browser_ms - float(trial["browser_onset_ms"])) * 1e6
                ),
                "frame_interval_ns": interval_ns, "late_by_ns": late_by_ns,
                "raw_events_consumed": int(frame.get("raw_events", 0)),
                "cursor_x_m": float(frame["cursor_x_m"]),
                "cursor_y_m": float(frame["cursor_y_m"]),
                "target_x_m": trial["template"].target_x_m,
                "target_y_m": trial["template"].target_y_m,
                "target_radius_m": trial["template"].target_radius_m,
                "distance_to_target_m": distance,
                "inside_target": distance <= trial["template"].target_radius_m,
                "window_focused": bool(frame.get("focused", True)),
                "browser_timestamp_ms": browser_ms,
                "browser_interval_ms": interval_ms,
                "browser_time_origin_ms": client.get("time_origin_ms"),
                "document_visible": bool(frame.get("visible", True)),
                "pointer_locked": bool(frame.get("pointer_locked", False)),
                "raw_event_sequence": int(frame.get("raw_sequence", 0)),
            })
        template = trial["template"]
        click_pos = trial["click_pos"]
        click_error = None if click_pos is None else float(np.linalg.norm(click_pos - trial["target"]))
        first_move = trial["first_movement_ns"]
        click_ns = trial["click_ns"]
        rendered_radius_value = client.get("rendered_target_radius_m")
        rendered_radius_m = (
            template.target_radius_m if rendered_radius_value is None
            else float(rendered_radius_value)
        )
        rendered_id = math.log2(template.target_distance_m / (2 * rendered_radius_m) + 1)
        display_calibration = client.get("display_calibration") or self.display_calibration or {}
        screen_window = client.get("screen_window") or {}
        physical_cm = display_calibration.get("physical_size_cm") or [None, None]
        resolution_px = display_calibration.get("resolution_px") or [None, None]
        px_per_m = display_calibration.get("px_per_m") or [None, None]
        summary = {
            "session_id": self.session_id, "trial_id": trial["trial_id"],
            "template_id": template.template_id, "phase": trial["phase"],
            "block_index": trial["block_index"], "cpi": trial["cpi"],
            "input_backend": self.input_backend,
            "attempt": trial["attempt"], "valid": trial["valid"],
            "invalid_reason": "" if trial["valid"] else trial["outcome"],
            "outcome": trial["outcome"], "success": trial["outcome"] == "success",
            "timeout": trial["outcome"] == "timeout", "onset_ns": trial["onset_ns"],
            "first_movement_ns": first_move, "click_ns": click_ns, "end_ns": trial["end_ns"],
            "reaction_time_s": None if first_move is None else (first_move - trial["onset_ns"]) / 1e9,
            "movement_time_s": None if first_move is None or click_ns is None else (click_ns - first_move) / 1e9,
            "total_time_s": (trial["end_ns"] - trial["onset_ns"]) / 1e9,
            "cursor_start_x_m": float(trial["cursor_start"][0]),
            "cursor_start_y_m": float(trial["cursor_start"][1]),
            "target_x_m": template.target_x_m, "target_y_m": template.target_y_m,
            "target_radius_m": template.target_radius_m, "target_width_m": 2 * template.target_radius_m,
            "rendered_target_radius_px": client.get("rendered_target_radius_px"),
            "rendered_target_radius_m": rendered_radius_m, "rendered_fitts_id": rendered_id,
            "target_distance_m": template.target_distance_m, "fitts_id": template.fitts_id,
            "click_x_m": None if click_pos is None else float(click_pos[0]),
            "click_y_m": None if click_pos is None else float(click_pos[1]),
            "click_error_m": click_error, "minimum_target_distance_m": trial["minimum_distance"],
            "path_length_m": trial["path_length"],
            "straight_line_efficiency": None if trial["path_length"] == 0 else template.target_distance_m / trial["path_length"],
            "raw_event_count": len(trial["raw_rows"]), "frame_count": len(frame_rows),
            "late_frame_count": late_count, "max_frame_interval_ns": max_interval_ns,
            "focus_lost": not trial["valid"], "browser_onset_ms": trial["browser_onset_ms"],
            "browser_first_movement_ms": client.get("first_movement_ms"),
            "browser_click_ms": client.get("click_ms"), "browser_end_ms": client.get("end_ms"),
            "pointer_lock_raw_requested": client.get("pointer_lock_raw_requested"),
            "pointer_lock_raw_granted": client.get("pointer_lock_raw_granted"),
            "client_fullscreen": client.get("fullscreen"),
            "client_width_px": client.get("width_px"), "client_height_px": client.get("height_px"),
            "device_pixel_ratio": client.get("device_pixel_ratio"),
            "client_inner_width_css_px": screen_window.get("inner_width_css_px"),
            "client_inner_height_css_px": screen_window.get("inner_height_css_px"),
            "client_outer_width_css_px": screen_window.get("outer_width_css_px"),
            "client_outer_height_css_px": screen_window.get("outer_height_css_px"),
            "screen_width_css_px": screen_window.get("screen_width_css_px"),
            "screen_height_css_px": screen_window.get("screen_height_css_px"),
            "screen_avail_width_css_px": screen_window.get("screen_avail_width_css_px"),
            "screen_avail_height_css_px": screen_window.get("screen_avail_height_css_px"),
            "visual_viewport_width_css_px": screen_window.get("visual_viewport_width_css_px"),
            "visual_viewport_height_css_px": screen_window.get("visual_viewport_height_css_px"),
            "canvas_css_width_px": screen_window.get("canvas_css_width_px"),
            "canvas_css_height_px": screen_window.get("canvas_css_height_px"),
            "canvas_bounding_left_px": screen_window.get("canvas_bounding_left_px"),
            "canvas_bounding_top_px": screen_window.get("canvas_bounding_top_px"),
            "canvas_bounding_width_px": screen_window.get("canvas_bounding_width_px"),
            "canvas_bounding_height_px": screen_window.get("canvas_bounding_height_px"),
            "reported_screen_width_cm": physical_cm[0],
            "reported_screen_height_cm": physical_cm[1],
            "reported_resolution_width_px": resolution_px[0],
            "reported_resolution_height_px": resolution_px[1],
            "reported_refresh_hz": display_calibration.get("refresh_hz"),
            "display_diagonal_in": display_calibration.get("diagonal_in"),
            "fixed_profile_warning": display_calibration.get("fixed_profile_warning"),
            "physical_px_per_m_x": px_per_m[0],
            "physical_px_per_m_y": px_per_m[1],
            "render_scale_px_per_virtual_m": client.get("render_scale_px_per_virtual_m"),
            "rendered_task_width_px": client.get("rendered_task_width_px"),
            "rendered_task_height_px": client.get("rendered_task_height_px"),
            "rendered_task_width_physical_m": client.get("rendered_task_width_physical_m"),
            "rendered_task_height_physical_m": client.get("rendered_task_height_physical_m"),
            "rendered_target_radius_physical_m": client.get("rendered_target_radius_physical_m"),
        }
        assert self.logger is not None
        self.logger.submit_trial(summary, trial["raw_rows"], frame_rows, trial["events"])
        trial_id = trial["trial_id"]
        self.trial = None
        await self._send({"type": "trial_saved", "trial_id": trial_id})


def create_app(
    config: dict[str, Any],
    output_root: Path,
    input_backend: str = "auto",
) -> web.Application:
    server = WebExperimentServer(config, output_root, input_backend=input_backend)
    app = web.Application()
    app["experiment_server"] = server
    app.router.add_get("/", server.index)
    app.router.add_get("/api/health", server.health)
    app.router.add_get("/ws", server.websocket)
    app.router.add_static("/static", STATIC_DIR, show_index=False)
    app.on_startup.append(server.start)
    app.on_cleanup.append(server.stop)
    return app


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/human_experiment")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--input-backend",
        choices=("auto", BACKEND_WINDOWS_RAW_INPUT, BACKEND_BROWSER_POINTER_LOCK),
        default="auto",
        help=(
            "auto uses Windows Raw Input on Windows and browser Pointer Lock on "
            "macOS/Linux. Use browser_pointer_lock for cross-platform pilot runs."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Point-and-click experiment: {url}")
    print(f"Input backend: {resolve_input_backend(args.input_backend)}")
    print("Keep this window open during data collection. Press Ctrl+C to stop.")
    web.run_app(
        create_app(config, args.output_root, input_backend=args.input_backend),
        host=args.host,
        port=args.port,
        print=None,
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
