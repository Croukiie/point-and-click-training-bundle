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
from .task_design import (
    angle_from_delta,
    counterbalanced_cpi_order,
    make_trial_stage_template,
    make_trial_templates,
    signed_angle_error,
)
from ..utils.mouse_sensitivity import (
    DEFAULT_MOUSE_REFERENCE_CPI,
    cursor_meters_per_count,
    physical_meters_per_count,
    positive_cpi,
)
from ..utils.trajectory_features import cursor_curvature_features

RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
try:
    if sys.platform != "win32":
        raise ImportError("Windows Raw Input is unavailable on this platform")
    from .raw_input_win32 import (
        RawInputMouse,
        get_os_cursor_position,
        set_os_cursor_position,
    )
except (ImportError, AttributeError):
    RawInputMouse = None
    get_os_cursor_position = None
    set_os_cursor_position = None


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).with_name("web")
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
BACKEND_WINDOWS_RAW_INPUT = "windows_raw_input"
BACKEND_BROWSER_POINTER_LOCK = "browser_pointer_lock"
CURSOR_PRESENTATION_MODES = {"virtual_raw_input", "native_windows"}

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


def normalize_native_screen_mapping(value: Any) -> dict[str, float | str]:
    """Validate one browser-to-Windows cursor coordinate transform."""
    if not isinstance(value, dict):
        raise ValueError("Native cursor mapping is missing")
    numeric = {}
    for name in (
        "screen_origin_x_px",
        "screen_origin_y_px",
        "task_offset_x_px",
        "task_offset_y_px",
        "task_scale_px_per_m",
    ):
        number = float(value.get(name))
        if not math.isfinite(number):
            raise ValueError(f"Native cursor mapping {name} must be finite")
        numeric[name] = number
    if numeric["task_scale_px_per_m"] <= 0:
        raise ValueError("Native cursor mapping scale must be positive")
    return {"basis": str(value.get("basis", "unknown")), **numeric}


def map_os_cursor_to_virtual(
    mapping: dict[str, float | str],
    os_cursor_x_px: int,
    os_cursor_y_px: int,
    virtual_size_m: tuple[float, float],
) -> np.ndarray:
    """Map an absolute Windows cursor position into the experiment space."""
    width_m, height_m = virtual_size_m
    scale = float(mapping["task_scale_px_per_m"])
    canvas_x = float(os_cursor_x_px) - float(mapping["screen_origin_x_px"])
    canvas_y = float(os_cursor_y_px) - float(mapping["screen_origin_y_px"])
    x_m = (canvas_x - float(mapping["task_offset_x_px"])) / scale - width_m / 2
    y_m = height_m / 2 - (canvas_y - float(mapping["task_offset_y_px"])) / scale
    return np.array(
        [
            np.clip(x_m, -width_m / 2, width_m / 2),
            np.clip(y_m, -height_m / 2, height_m / 2),
        ],
        dtype=np.float64,
    )


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


def normalize_initial_cursor(
    value: Any,
    virtual_size_m: tuple[float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = value if isinstance(value, dict) else {}
    try:
        cursor_x = float(payload.get("cursor_x_m", 0.0))
        cursor_y = float(payload.get("cursor_y_m", 0.0))
    except (TypeError, ValueError):
        cursor_x, cursor_y = 0.0, 0.0
    if not math.isfinite(cursor_x) or not math.isfinite(cursor_y):
        cursor_x, cursor_y = 0.0, 0.0
    half_width = float(virtual_size_m[0]) / 2
    half_height = float(virtual_size_m[1]) / 2
    cursor = np.array(
        [
            np.clip(cursor_x, -half_width, half_width),
            np.clip(cursor_y, -half_height, half_height),
        ],
        dtype=np.float64,
    )
    metadata = {
        **payload,
        "source": str(payload.get("source", "center_fallback")),
        "cursor_x_m": float(cursor[0]),
        "cursor_y_m": float(cursor[1]),
        "clipped_to_virtual_screen": bool(
            float(cursor[0]) != cursor_x or float(cursor[1]) != cursor_y
        ),
    }
    return cursor, metadata


class WebExperimentServer:
    def __init__(
        self,
        config: dict[str, Any],
        output_root: Path,
        input_backend: str = "auto",
        mousepad_condition: str = "unspecified",
    ):
        self.config = config
        self.exp = config["experiment"]
        self.mouse_reference_cpi = positive_cpi(
            self.exp.get("mouse_reference_cpi", DEFAULT_MOUSE_REFERENCE_CPI),
            name="experiment.mouse_reference_cpi",
        )
        self.cursor_meters_per_count = cursor_meters_per_count(
            self.exp["mouse_gain"],
            self.mouse_reference_cpi,
        )
        self.output_root = output_root
        self.input_backend = resolve_input_backend(input_backend)
        self.mousepad_condition = mousepad_condition
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
        self.active_cpi: int | None = None
        self.cpi_setting_method = "not_reported"
        self.cursor_presentation_mode = "virtual_raw_input"
        self.native_screen_mapping: dict[str, float | str] | None = None
        self.cpi_confirmed_blocks: dict[int, dict[str, Any]] = {}
        self.templates = []
        self.template_by_id = {}
        self.trial: dict[str, Any] | None = None
        self.trial_sequence = 0
        self.display_calibration: dict[str, Any] | None = None
        self.current_cursor = np.zeros(2, dtype=np.float64)
        self.current_raw_session_pos = np.zeros(2, dtype=np.float64)
        self.intertrial_motion_enabled = False
        self.intertrial_motion: dict[str, Any] | None = None

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
        self.active_cpi = None
        self.cpi_setting_method = "not_reported"
        self.cursor_presentation_mode = "virtual_raw_input"
        self.native_screen_mapping = None
        self.cpi_confirmed_blocks = {}
        self.templates = []
        self.template_by_id = {}
        self.trial = None
        self.trial_sequence = 0
        self.display_calibration = None
        self.current_cursor = np.zeros(2, dtype=np.float64)
        self.current_raw_session_pos = np.zeros(2, dtype=np.float64)
        self.intertrial_motion_enabled = False
        self.intertrial_motion = None
        if self.timeout_task:
            self.timeout_task.cancel()
            self.timeout_task = None
        return session_id

    async def index(self, request: web.Request):
        return web.FileResponse(
            STATIC_DIR / "index.html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )

    async def health(self, request: web.Request):
        return web.json_response({
            "status": "ok",
            "input_backend": self.input_backend,
            "raw_input": self.input_backend == BACKEND_WINDOWS_RAW_INPUT,
            "session_active": self.logger is not None,
            "protocol_version": self.exp.get("protocol_version", "unversioned"),
            "start_target_enabled": bool(
                self.exp.get("start_target_enabled", False)
            ),
            "practice_trials_per_cpi": int(self.exp["practice_trials_per_cpi"]),
            "measured_trials_per_cpi": int(self.exp["measured_trials_per_cpi"]),
            "cpi_condition_count": len(self.exp["cpi_values"]),
            "ab_pilot": bool(self.exp.get("cursor_ab_pilot", False)),
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
        elif message_type == "confirm_cpi":
            await self._confirm_cpi(data)
        elif message_type == "target_onset":
            await self._target_onset(data)
        elif message_type == "client_input_batch":
            await self._client_input_batch(data)
        elif message_type == "intertrial_client_input":
            await self._client_intertrial_input(data)
        elif message_type == "set_initial_cursor":
            await self._set_initial_cursor(data)
        elif message_type == "commit_overlay_cursor":
            await self._commit_overlay_cursor(data)
        elif message_type == "set_intertrial_motion":
            if data.get("enabled") is False:
                self._finalize_intertrial_motion(
                    next_trial_id=None,
                    reason=str(data.get("reason", "client_disabled")),
                )
        elif message_type == "sync_os_cursor_to_virtual":
            await self._sync_os_cursor_to_virtual(data)
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
                    "trial_stage": data.get("trial_stage"),
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
        declared_polling_hz = int(
            hardware.get(
                "declared_mouse_polling_hz",
                self.config["logging"]["expected_mouse_polling_hz"],
            )
        )
        if declared_polling_hz <= 0:
            raise ValueError("Effective Raw Input rate must be positive")
        hardware["declared_mouse_polling_hz"] = declared_polling_hz
        cpi_setting_method = str(
            hardware.get("cpi_setting_method", "not_reported")
        ).strip()
        allowed_cpi_methods = {
            "physical_button", "mouse_software", "hardware_profile", "other"
        }
        if cpi_setting_method not in allowed_cpi_methods:
            raise ValueError("Select a valid CPI setting method")
        self.cpi_setting_method = cpi_setting_method
        self.cpi_confirmed_blocks = {}
        hardware["cpi_setting_method"] = cpi_setting_method
        cursor_presentation_mode = str(
            hardware.get("cursor_presentation_mode", "virtual_raw_input")
        ).strip()
        if cursor_presentation_mode not in CURSOR_PRESENTATION_MODES:
            raise ValueError("Select a valid cursor presentation mode")
        if (
            cursor_presentation_mode == "native_windows"
            and self.input_backend != BACKEND_WINDOWS_RAW_INPUT
        ):
            raise ValueError("The native Windows cursor condition requires Windows Raw Input")
        self.cursor_presentation_mode = cursor_presentation_mode
        hardware["cursor_presentation_mode"] = cursor_presentation_mode
        self.display_calibration = normalize_display_calibration(
            hardware.get("display_calibration")
        )
        validate_fixed_display_profile(self.display_calibration, self.config["display"])
        hardware["display_calibration"] = self.display_calibration
        hardware["declared_resolution_px"] = self.display_calibration["resolution_px"]
        hardware["declared_refresh_hz"] = self.display_calibration["refresh_hz"]
        hardware["mousepad_condition"] = self.mousepad_condition
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
        start_target_enabled = bool(self.exp.get("start_target_enabled", False))
        self.templates = make_trial_templates(
            count, virtual_size, tuple(float(v) for v in self.exp["target_id_range"]),
            int(self.exp["seed"]),
            cursor_position_m=None,
            target_radius_range_m=tuple(
                float(v)
                for v in self.exp.get("target_radius_range_m", (0.00125, 0.007))
            ),
        )
        self.template_by_id = {item.template_id: item for item in self.templates}
        self.current_cursor, initial_cursor_metadata = normalize_initial_cursor(
            data.get("initial_cursor"), virtual_size
        )
        self.current_raw_session_pos = np.zeros(2, dtype=np.float64)
        self.active_cpi = None
        self.intertrial_motion_enabled = False
        self.intertrial_motion = None
        output_dir = self.output_root / self.session_id
        clock = time.get_clock_info("perf_counter")
        metadata = {
            "schema_version": 2,
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
                "initial_cursor": initial_cursor_metadata,
                "intertrial_cursor_motion": (
                    "Raw movement between trials is logged separately and sets "
                    "the cursor start position of the next trial. It is excluded "
                    "from the completed trial trajectory and its derived features."
                ),
                "cpi_confirmation_required": True,
                "start_target_enabled": start_target_enabled,
                "start_target_radius_m": float(self.exp.get("start_target_radius_m", 0.006)),
                "trial_stage_note": (
                    "Reference/start targets are disabled. Each target trial "
                    "starts from the current virtual cursor position after the "
                    "participant's separately logged inter-trial movement."
                    if not start_target_enabled
                    else "Each logical trial is logged as a sampled start "
                    "target followed by its jointly sampled task target. After "
                    "a successful start click, the virtual cursor is snapped "
                    "to the start-target center so simulator geometry is exact."
                ),
                "cursor_source": (
                    "Windows GetCursorPos mapped to the fullscreen task; Raw Input counts saved separately"
                    if cursor_presentation_mode == "native_windows"
                    else (
                        "Windows Raw Input reconstructed by Python"
                        if self.input_backend == BACKEND_WINDOWS_RAW_INPUT
                        else "Browser Pointer Lock movement reconstructed by the local web client"
                    )
                ),
                "cursor_presentation_mode": cursor_presentation_mode,
                "native_cursor_mapping_version": (
                    "getcursorpos_dual_dpi_auto_v2"
                    if cursor_presentation_mode == "native_windows" else None
                ),
                "cursor_presentation_note": (
                    "The native operating-system cursor was visible. Windows "
                    "GetCursorPos was mapped into task coordinates for the saved "
                    "visible trajectory and hit test. Raw Input dx/dy counts and "
                    "their reconstructed trajectory were saved separately."
                    if cursor_presentation_mode == "native_windows"
                    else "The controlled virtual aero-arrow cursor was rendered "
                    "from the Python Windows Raw Input reconstruction."
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
            "cursor_presentation_mode": cursor_presentation_mode,
        })
        await self._send({
            "type": "session_ready", "session_id": self.session_id,
            "cpi_order": self.cpi_order,
            "mousepad_condition": self.mousepad_condition,
            "templates": [item.to_dict() for item in self.templates],
            "experiment": self.exp,
            "display": self.config["display"],
            "display_calibration": self.display_calibration,
            "appearance": self.config["appearance"],
            "input_backend": self.input_backend,
            "cursor_x_m": float(self.current_cursor[0]),
            "cursor_y_m": float(self.current_cursor[1]),
        })

    async def _confirm_cpi(self, data: dict[str, Any]):
        if self.logger is None:
            raise ValueError("Start a session before confirming CPI")
        block_index = int(data["block_index"])
        cpi = int(data["cpi"])
        if block_index < 1 or block_index > len(self.cpi_order):
            raise ValueError("Invalid CPI block index")
        if cpi != self.cpi_order[block_index - 1]:
            raise ValueError("CPI confirmation does not match the current block")
        method = str(data.get("cpi_setting_method", "")).strip()
        if method != self.cpi_setting_method:
            raise ValueError("CPI setting method changed after session start")
        confirmation = {
            "cpi": cpi,
            "cpi_setting_method": method,
            "confirmed_server_ns": time.perf_counter_ns(),
            "confirmed_browser_ms": data.get("browser_ms"),
        }
        self.cpi_confirmed_blocks[block_index] = confirmation
        self.active_cpi = cpi
        self.logger.submit_event({
            "event": "cpi_confirmed",
            "session_id": self.session_id,
            "timestamp_ns": confirmation["confirmed_server_ns"],
            "block_index": block_index,
            **confirmation,
        })
        await self._send({
            "type": "cpi_confirmation_saved",
            "block_index": block_index,
            "cpi": cpi,
        })

    def _enable_intertrial_motion(self, trial: dict[str, Any]) -> None:
        """Start a cursor-only segment after a completed task trial."""
        if trial.get("trial_stage") != "task" or self.active_cpi is None:
            self.intertrial_motion_enabled = False
            self.intertrial_motion = None
            return
        self.current_cursor = trial["cursor"].copy()
        self.current_raw_session_pos = trial["raw_session_pos"].copy()
        now_ns = time.perf_counter_ns()
        self.intertrial_motion_enabled = True
        self.intertrial_motion = {
            "start_ns": now_ns,
            "end_ns": now_ns,
            "event_count": 0,
            "button_down_count": 0,
            "dx_counts": 0,
            "dy_counts": 0,
            "cpi": int(self.active_cpi),
            "previous_trial_id": int(trial["trial_id"]),
            "phase": trial.get("phase"),
            "block_index": trial.get("block_index"),
            "cursor_start_x_m": float(self.current_cursor[0]),
            "cursor_start_y_m": float(self.current_cursor[1]),
            "raw_session_start_x_m": float(self.current_raw_session_pos[0]),
            "raw_session_start_y_m": float(self.current_raw_session_pos[1]),
        }

    def _apply_intertrial_input(
        self,
        *,
        timestamp_ns: int,
        dx_counts: int,
        dy_counts: int,
        button_flags: int = 0,
        cursor_x_m: float | None = None,
        cursor_y_m: float | None = None,
        os_cursor_x_px: int | None = None,
        os_cursor_y_px: int | None = None,
    ) -> bool:
        if (
            not self.intertrial_motion_enabled
            or self.intertrial_motion is None
            or self.active_cpi is None
        ):
            return False
        virtual_width, virtual_height = (
            float(value) for value in self.config["display"]["virtual_size_m"]
        )
        native_cursor = self._native_cursor_position(os_cursor_x_px, os_cursor_y_px)
        if native_cursor is not None:
            self.current_cursor = native_cursor
        elif cursor_x_m is None or cursor_y_m is None:
            self.current_cursor[0] += int(dx_counts) * self.cursor_meters_per_count
            self.current_cursor[1] -= int(dy_counts) * self.cursor_meters_per_count
        else:
            self.current_cursor[0] = float(cursor_x_m)
            self.current_cursor[1] = float(cursor_y_m)
        self.current_cursor[0] = np.clip(
            self.current_cursor[0], -virtual_width / 2, virtual_width / 2
        )
        self.current_cursor[1] = np.clip(
            self.current_cursor[1], -virtual_height / 2, virtual_height / 2
        )
        raw_m_per_count = physical_meters_per_count(self.active_cpi)
        self.current_raw_session_pos += np.array(
            [int(dx_counts) * raw_m_per_count, -int(dy_counts) * raw_m_per_count],
            dtype=np.float64,
        )
        motion = self.intertrial_motion
        motion["end_ns"] = int(timestamp_ns)
        motion["event_count"] += 1
        motion["dx_counts"] += int(dx_counts)
        motion["dy_counts"] += int(dy_counts)
        if int(button_flags) & RI_MOUSE_LEFT_BUTTON_DOWN:
            motion["button_down_count"] += 1
        return True

    def _finalize_intertrial_motion(
        self,
        *,
        next_trial_id: int | None,
        reason: str,
    ) -> None:
        motion = self.intertrial_motion
        if motion is not None and motion["event_count"] > 0 and self.logger is not None:
            self.logger.submit_event({
                "event": "intertrial_movement",
                "session_id": self.session_id,
                **motion,
                "next_trial_id": next_trial_id,
                "reason": reason,
                "duration_s": max(0.0, (motion["end_ns"] - motion["start_ns"]) / 1e9),
                "cursor_end_x_m": float(self.current_cursor[0]),
                "cursor_end_y_m": float(self.current_cursor[1]),
                "raw_session_end_x_m": float(self.current_raw_session_pos[0]),
                "raw_session_end_y_m": float(self.current_raw_session_pos[1]),
                "input_backend": self.input_backend,
            })
        self.intertrial_motion_enabled = False
        self.intertrial_motion = None

    async def _client_intertrial_input(self, data: dict[str, Any]) -> None:
        if self.input_backend != BACKEND_BROWSER_POINTER_LOCK:
            return
        applied = self._apply_intertrial_input(
            timestamp_ns=time.perf_counter_ns(),
            dx_counts=int(round(float(data.get("dx_counts", 0)))),
            dy_counts=int(round(float(data.get("dy_counts", 0)))),
            cursor_x_m=data.get("cursor_x_m"),
            cursor_y_m=data.get("cursor_y_m"),
        )
        if applied:
            await self._send({
                "type": "intertrial_cursor",
                "cursor_x_m": float(self.current_cursor[0]),
                "cursor_y_m": float(self.current_cursor[1]),
            })

    async def _set_initial_cursor(self, data: dict[str, Any]) -> None:
        """Position the cursor once, from the first fullscreen start click."""
        if self.logger is None or self.trial is not None or self.trial_sequence != 0:
            return
        previous_cursor = self.current_cursor.copy()
        cursor, cursor_metadata = normalize_initial_cursor(
            {
                "source": data.get("source", "first_fullscreen_start_button"),
                "cursor_x_m": data.get("cursor_x_m"),
                "cursor_y_m": data.get("cursor_y_m"),
            },
            tuple(float(value) for value in self.config["display"]["virtual_size_m"]),
        )
        self.current_cursor = cursor
        self.logger.submit_event({
            "event": "initial_virtual_cursor_positioned",
            "session_id": self.session_id,
            "timestamp_ns": time.perf_counter_ns(),
            "browser_ms": data.get("browser_ms"),
            "source": cursor_metadata["source"],
            "previous_cursor_x_m": float(previous_cursor[0]),
            "previous_cursor_y_m": float(previous_cursor[1]),
            "cursor_x_m": float(cursor[0]),
            "cursor_y_m": float(cursor[1]),
            "input_backend": self.input_backend,
        })

    def _select_native_screen_mapping(
        self,
        payload: Any,
        browser_cursor: np.ndarray,
    ) -> tuple[dict[str, float | str] | None, dict[str, Any]]:
        """Choose logical or physical pixels using the live OS cursor position."""
        if getattr(self, "cursor_presentation_mode", "virtual_raw_input") != "native_windows":
            return None, {}
        if not isinstance(payload, dict):
            raise ValueError("Native Windows cursor mapping was not provided")
        candidates = []
        for basis in ("logical", "physical"):
            if basis not in payload:
                continue
            candidate = normalize_native_screen_mapping(payload[basis])
            candidate["basis"] = basis
            candidates.append(candidate)
        if not candidates:
            raise ValueError("No valid native Windows cursor mapping was provided")
        os_position = get_os_cursor_position() if get_os_cursor_position else None
        if os_position is None:
            selected = next(
                (item for item in candidates if item["basis"] == "physical"),
                candidates[0],
            )
            return selected, {
                "selection_method": "physical_fallback_no_os_position",
                "selection_error_m": None,
                "os_cursor_x_px": None,
                "os_cursor_y_px": None,
            }
        virtual_size = tuple(
            float(value) for value in self.config["display"]["virtual_size_m"]
        )
        scored = []
        for candidate in candidates:
            mapped = map_os_cursor_to_virtual(
                candidate, os_position[0], os_position[1], virtual_size
            )
            scored.append((float(np.linalg.norm(mapped - browser_cursor)), candidate, mapped))
        error, selected, mapped = min(scored, key=lambda item: item[0])
        return selected, {
            "selection_method": "closest_to_browser_click",
            "selection_error_m": error,
            "os_cursor_x_px": int(os_position[0]),
            "os_cursor_y_px": int(os_position[1]),
            "mapped_cursor_x_m": float(mapped[0]),
            "mapped_cursor_y_m": float(mapped[1]),
        }

    def _native_cursor_position(
        self,
        os_cursor_x_px: int | None,
        os_cursor_y_px: int | None,
    ) -> np.ndarray | None:
        if (
            getattr(self, "cursor_presentation_mode", "virtual_raw_input") != "native_windows"
            or getattr(self, "native_screen_mapping", None) is None
            or os_cursor_x_px is None
            or os_cursor_y_px is None
        ):
            return None
        return map_os_cursor_to_virtual(
            self.native_screen_mapping,
            int(os_cursor_x_px),
            int(os_cursor_y_px),
            tuple(float(v) for v in self.config["display"]["virtual_size_m"]),
        )

    async def _commit_overlay_cursor(self, data: dict[str, Any]) -> None:
        """Commit the custom overlay cursor as the next trial's start position."""
        if self.logger is None:
            raise ValueError("Start a session before committing the cursor")
        if self.trial is not None and self.trial.get("state") == "active":
            raise ValueError("The cursor cannot be committed during an active trial")

        self._finalize_intertrial_motion(
            next_trial_id=None,
            reason="overlay_cursor_commit",
        )
        if self.raw_mouse is not None:
            self.raw_mouse.drain()

        previous_cursor = self.current_cursor.copy()
        cursor, cursor_metadata = normalize_initial_cursor(
            {
                "source": data.get("source", "overlay_cursor_click"),
                "cursor_x_m": data.get("cursor_x_m"),
                "cursor_y_m": data.get("cursor_y_m"),
            },
            tuple(float(value) for value in self.config["display"]["virtual_size_m"]),
        )
        mapping_details: dict[str, Any] = {}
        cursor_presentation_mode = getattr(
            self, "cursor_presentation_mode", "virtual_raw_input"
        )
        if cursor_presentation_mode == "native_windows":
            self.native_screen_mapping, mapping_details = self._select_native_screen_mapping(
                data.get("native_screen_mappings"), cursor
            )
            if mapping_details.get("mapped_cursor_x_m") is not None:
                cursor = np.array(
                    [
                        mapping_details["mapped_cursor_x_m"],
                        mapping_details["mapped_cursor_y_m"],
                    ],
                    dtype=np.float64,
                )
        self.current_cursor = cursor
        self.logger.submit_event({
            "event": "overlay_cursor_committed",
            "session_id": self.session_id,
            "timestamp_ns": time.perf_counter_ns(),
            "browser_ms": data.get("browser_ms"),
            "request_id": str(data.get("request_id", "")),
            "source": cursor_metadata["source"],
            "previous_cursor_x_m": float(previous_cursor[0]),
            "previous_cursor_y_m": float(previous_cursor[1]),
            "cursor_x_m": float(cursor[0]),
            "cursor_y_m": float(cursor[1]),
            "clipped_to_virtual_screen": cursor_metadata["clipped_to_virtual_screen"],
            "input_backend": self.input_backend,
            "cursor_presentation_mode": cursor_presentation_mode,
            "native_screen_mapping": getattr(self, "native_screen_mapping", None),
            **mapping_details,
        })
        await self._send({
            "type": "overlay_cursor_committed",
            "request_id": str(data.get("request_id", "")),
            "cursor_x_m": float(cursor[0]),
            "cursor_y_m": float(cursor[1]),
        })

    async def _sync_os_cursor_to_virtual(self, data: dict[str, Any]) -> None:
        if (
            self.input_backend != BACKEND_WINDOWS_RAW_INPUT
            or set_os_cursor_position is None
            or self.display_calibration is None
        ):
            return
        if self.trial is not None and self.trial.get("state") == "active":
            return
        resolution_width, resolution_height = (
            int(value) for value in self.display_calibration["resolution_px"]
        )
        virtual_width, virtual_height = (
            float(value) for value in self.config["display"]["virtual_size_m"]
        )
        try:
            origin_x = int(round(float(data.get("screen_origin_x_px", 0))))
            origin_y = int(round(float(data.get("screen_origin_y_px", 0))))
        except (TypeError, ValueError):
            origin_x, origin_y = 0, 0
        x_fraction = np.clip(self.current_cursor[0] / virtual_width + 0.5, 0.0, 1.0)
        y_fraction = np.clip(0.5 - self.current_cursor[1] / virtual_height, 0.0, 1.0)
        x_px = origin_x + round(float(x_fraction) * max(0, resolution_width - 1))
        y_px = origin_y + round(float(y_fraction) * max(0, resolution_height - 1))
        positioned = set_os_cursor_position(x_px, y_px)
        if self.logger is not None:
            self.logger.submit_event({
                "event": "os_cursor_synced_to_virtual",
                "session_id": self.session_id,
                "timestamp_ns": time.perf_counter_ns(),
                "reason": str(data.get("reason", "pointer_lock_exit")),
                "cursor_x_m": float(self.current_cursor[0]),
                "cursor_y_m": float(self.current_cursor[1]),
                "os_cursor_x_px": x_px,
                "os_cursor_y_px": y_px,
                "positioned": positioned,
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
        block_index = int(data["block_index"])
        confirmation = self.cpi_confirmed_blocks.get(block_index)
        if (
            data.get("cpi_confirmed") is not True
            or confirmation is None
            or int(confirmation["cpi"]) != cpi
        ):
            raise ValueError("Confirm the requested CPI setting before starting this block")
        cpi_setting_method = str(data.get("cpi_setting_method", "")).strip()
        if cpi_setting_method != self.cpi_setting_method:
            raise ValueError("CPI setting method does not match the session metadata")
        phase = str(data["phase"])
        if phase not in {"practice", "measured"}:
            raise ValueError("Invalid trial phase")
        trial_stage = str(data.get("trial_stage", "task"))
        if trial_stage not in {"start", "task"}:
            raise ValueError("Invalid trial stage")
        if not bool(self.exp.get("start_target_enabled", False)):
            trial_stage = "task"
        base_template = self.template_by_id[template_id]
        self._finalize_intertrial_motion(
            next_trial_id=self.trial_sequence + 1,
            reason="next_trial_started",
        )
        self.trial_sequence += 1
        cursor = self.current_cursor.copy()
        template = make_trial_stage_template(
            base_template,
            trial_stage,
            cursor,
            tuple(float(v) for v in self.config["display"]["virtual_size_m"]),
            float(self.exp.get("start_target_radius_m", 0.006)),
            tuple(float(v) for v in self.exp["target_radius_range_m"]),
            preserve_sampled_task_geometry=(
                self.exp.get("protocol_version") == "simulator_geometry_exact_v3"
            ),
        )
        target = np.array([template.target_x_m, template.target_y_m], dtype=np.float64)
        if self.raw_mouse is not None:
            self.raw_mouse.drain()
        self.trial = {
            "state": "ready", "trial_id": self.trial_sequence,
            "template": template, "base_template": base_template,
            "phase": phase, "trial_stage": trial_stage, "cpi": cpi,
            "block_index": block_index, "attempt": int(data.get("attempt", 1)),
            "cpi_setting_method": cpi_setting_method, "cpi_confirmed": True,
            "cursor": cursor, "cursor_start": cursor.copy(), "target": target,
            "raw_reconstructed_cursor": cursor.copy(),
            "native_raw_alignment_errors": [],
            "path_length": 0.0, "minimum_distance": float(np.linalg.norm(cursor - target)),
            "first_movement_ns": None, "previous_raw_ns": None,
            "previous_cursor_velocity": None, "previous_raw_velocity": None,
            "raw_trial_pos": np.zeros(2, dtype=np.float64),
            "raw_session_start": self.current_raw_session_pos.copy(),
            "raw_session_pos": self.current_raw_session_pos.copy(),
            "raw_path_length": 0.0, "raw_max_speed": 0.0,
            "raw_dt_ns_values": [],
            "raw_rows": [], "events": [], "event_index": 0,
            "click_ns": None, "click_pos": None, "onset_ns": None,
            "click_raw_trial_pos": None, "click_raw_session_pos": None,
            "browser_onset_ms": None,
            "input_backend": self.input_backend,
        }
        await self._send({
            "type": "trial_ready", "trial_id": self.trial_sequence,
            "trial_stage": trial_stage,
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
            "phase": self.trial["phase"],
            "trial_stage": self.trial["trial_stage"],
            "browser_ms": self.trial["browser_onset_ms"],
        })
        await self._send({
            "type": "trial_active", "trial_id": self.trial["trial_id"],
            "server_onset_ns": onset_ns,
        })
        if (
            self.input_backend == BACKEND_BROWSER_POINTER_LOCK
            and self.trial["trial_stage"] != "start"
        ):
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
            and trial.get("trial_stage") != "start"
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
        raw_reconstructed_delta = np.array(
            [
                float(dx_counts) * self.cursor_meters_per_count,
                -float(dy_counts) * self.cursor_meters_per_count,
            ],
            dtype=np.float64,
        )
        trial["raw_reconstructed_cursor"] += raw_reconstructed_delta
        trial["raw_reconstructed_cursor"][0] = np.clip(
            trial["raw_reconstructed_cursor"][0], -virtual_width / 2, virtual_width / 2
        )
        trial["raw_reconstructed_cursor"][1] = np.clip(
            trial["raw_reconstructed_cursor"][1], -virtual_height / 2, virtual_height / 2
        )
        native_cursor = self._native_cursor_position(os_cursor_x_px, os_cursor_y_px)

        if native_cursor is not None:
            trial["cursor"] = native_cursor
        elif cursor_x_m is not None and cursor_y_m is not None:
            trial["cursor"][0] = float(cursor_x_m)
            trial["cursor"][1] = float(cursor_y_m)
        else:
            trial["cursor"] = trial["raw_reconstructed_cursor"].copy()

        trial["cursor"][0] = np.clip(trial["cursor"][0], -virtual_width / 2, virtual_width / 2)
        trial["cursor"][1] = np.clip(trial["cursor"][1], -virtual_height / 2, virtual_height / 2)

        movement = trial["cursor"] - old_cursor
        movement_norm = float(np.linalg.norm(movement))
        trial["path_length"] += movement_norm
        if movement_norm > 0 and trial["first_movement_ns"] is None:
            trial["first_movement_ns"] = timestamp_ns
        previous_ns = trial["previous_raw_ns"]
        dt_ns = 0 if previous_ns is None else max(0, timestamp_ns - previous_ns)
        if previous_ns is not None:
            trial["raw_dt_ns_values"].append(dt_ns)
        dt_s = dt_ns / 1e9 if dt_ns else 0.0
        velocity = movement / (dt_ns / 1e9) if dt_ns else np.zeros(2)
        previous_cursor_velocity = trial.get("previous_cursor_velocity")
        cursor_acceleration = (
            (velocity - previous_cursor_velocity) / dt_s
            if previous_cursor_velocity is not None and dt_s > 0
            else np.zeros(2)
        )

        raw_m_per_count = physical_meters_per_count(trial["cpi"])
        raw_delta = np.array(
            [float(dx_counts) * raw_m_per_count, -float(dy_counts) * raw_m_per_count],
            dtype=np.float64,
        )
        trial["raw_trial_pos"] = trial["raw_trial_pos"] + raw_delta
        trial["raw_session_pos"] = trial["raw_session_pos"] + raw_delta
        raw_movement_norm = float(np.linalg.norm(raw_delta))
        trial["raw_path_length"] += raw_movement_norm
        raw_velocity = raw_delta / dt_s if dt_s > 0 else np.zeros(2)
        previous_raw_velocity = trial.get("previous_raw_velocity")
        raw_acceleration = (
            (raw_velocity - previous_raw_velocity) / dt_s
            if previous_raw_velocity is not None and dt_s > 0
            else np.zeros(2)
        )
        raw_speed = float(np.linalg.norm(raw_velocity))
        trial["raw_max_speed"] = max(float(trial["raw_max_speed"]), raw_speed)

        distance = float(np.linalg.norm(trial["cursor"] - trial["target"]))
        trial["minimum_distance"] = min(trial["minimum_distance"], distance)
        cursor_heading = angle_from_delta(movement)
        target_vector = trial["target"] - trial["cursor"]
        target_angle = angle_from_delta(target_vector)
        raw_angle = angle_from_delta(raw_delta)
        heading_error = signed_angle_error(cursor_heading, target_angle)
        inside = distance <= trial["template"].target_radius_m
        native_raw_alignment_error = (
            float(np.linalg.norm(trial["cursor"] - trial["raw_reconstructed_cursor"]))
            if native_cursor is not None else None
        )
        if native_raw_alignment_error is not None:
            trial["native_raw_alignment_errors"].append(native_raw_alignment_error)
        trial["event_index"] += 1

        row = {
            "session_id": self.session_id, "trial_id": trial["trial_id"],
            "phase": trial["phase"], "trial_stage": trial["trial_stage"],
            "block_index": trial["block_index"],
            "cpi": trial["cpi"],
            "mousepad_condition": self.mousepad_condition,
            "input_backend": self.input_backend,
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
            "raw_dx_m": float(raw_delta[0]),
            "raw_dy_m": float(raw_delta[1]),
            "raw_trial_x_m": float(trial["raw_trial_pos"][0]),
            "raw_trial_y_m": float(trial["raw_trial_pos"][1]),
            "raw_session_x_m": float(trial["raw_session_pos"][0]),
            "raw_session_y_m": float(trial["raw_session_pos"][1]),
            "raw_path_length_m": float(trial["raw_path_length"]),
            "raw_displacement_from_trial_start_m": float(np.linalg.norm(trial["raw_trial_pos"])),
            "raw_speed_m_s": raw_speed,
            "raw_angle_rad": raw_angle,
            "raw_acceleration_x_m_s2": float(raw_acceleration[0]),
            "raw_acceleration_y_m_s2": float(raw_acceleration[1]),
            "raw_acceleration_m_s2": float(np.linalg.norm(raw_acceleration)),
            "cursor_delta_x_m": float(movement[0]),
            "cursor_delta_y_m": float(movement[1]),
            "cursor_heading_rad": cursor_heading,
            "cursor_path_length_m": float(trial["path_length"]),
            "cursor_x_m": float(trial["cursor"][0]),
            "cursor_y_m": float(trial["cursor"][1]),
            "raw_reconstructed_cursor_x_m": float(trial["raw_reconstructed_cursor"][0]),
            "raw_reconstructed_cursor_y_m": float(trial["raw_reconstructed_cursor"][1]),
            "native_raw_alignment_error_m": native_raw_alignment_error,
            "cursor_coordinate_source": (
                "windows_getcursorpos"
                if native_cursor is not None
                else "raw_input_reconstruction"
            ),
            "velocity_x_m_s": float(velocity[0]),
            "velocity_y_m_s": float(velocity[1]),
            "speed_m_s": float(np.linalg.norm(velocity)),
            "cursor_acceleration_x_m_s2": float(cursor_acceleration[0]),
            "cursor_acceleration_y_m_s2": float(cursor_acceleration[1]),
            "cursor_acceleration_m_s2": float(np.linalg.norm(cursor_acceleration)),
            "target_x_m": trial["template"].target_x_m,
            "target_y_m": trial["template"].target_y_m,
            "target_radius_m": trial["template"].target_radius_m,
            "distance_to_target_m": distance,
            "angle_to_target_rad": target_angle,
            "heading_error_rad": heading_error,
            "inside_target": inside,
        }
        trial["raw_rows"].append(row)
        trial["previous_raw_ns"] = timestamp_ns
        trial["previous_cursor_velocity"] = velocity.copy()
        trial["previous_raw_velocity"] = raw_velocity.copy()
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
                trial["click_raw_trial_pos"] = trial["raw_trial_pos"].copy()
                trial["click_raw_session_pos"] = trial["raw_session_pos"].copy()
                finish_outcome = "success" if inside else "miss"
                trial["events"].append({
                    "event": "click", "session_id": self.session_id,
                    "trial_id": trial["trial_id"], "phase": trial["phase"],
                    "trial_stage": trial["trial_stage"], "timestamp_ns": timestamp_ns,
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
                moved_between_trials = False
                for raw in raw_events:
                    moved_between_trials = self._apply_intertrial_input(
                        timestamp_ns=raw.timestamp_ns,
                        dx_counts=raw.dx,
                        dy_counts=raw.dy,
                        button_flags=raw.button_flags,
                        os_cursor_x_px=raw.os_cursor_x_px,
                        os_cursor_y_px=raw.os_cursor_y_px,
                    ) or moved_between_trials
                if moved_between_trials:
                    await self._send({
                        "type": "intertrial_cursor",
                        "cursor_x_m": float(self.current_cursor[0]),
                        "cursor_y_m": float(self.current_cursor[1]),
                    })
                continue
            onset_ns = trial["onset_ns"]
            batch = []
            finish_outcome = None
            for raw in raw_events:
                row, inside = self._append_pointer_event_row(
                    trial,
                    timestamp_ns=raw.timestamp_ns,
                    dx_counts=raw.dx,
                    dy_counts=raw.dy,
                    button_flags=raw.button_flags,
                    button_data=raw.button_data,
                    device_handle=raw.device_handle,
                    device_name=raw.device_name,
                    os_cursor_x_px=raw.os_cursor_x_px,
                    os_cursor_y_px=raw.os_cursor_y_px,
                    raw_flags=raw.raw_flags,
                )
                batch.append({
                    "sequence": trial["event_index"], "timestamp_ns": raw.timestamp_ns,
                    "cursor_x_m": row["cursor_x_m"], "cursor_y_m": row["cursor_y_m"],
                    "button_flags": raw.button_flags, "inside_target": inside,
                })
                if raw.button_flags & RI_MOUSE_LEFT_BUTTON_DOWN:
                    trial["click_ns"] = raw.timestamp_ns
                    trial["click_pos"] = trial["cursor"].copy()
                    trial["click_raw_trial_pos"] = trial["raw_trial_pos"].copy()
                    trial["click_raw_session_pos"] = trial["raw_session_pos"].copy()
                    finish_outcome = "success" if inside else "miss"
                    trial["events"].append({
                        "event": "click", "session_id": self.session_id,
                        "trial_id": trial["trial_id"], "phase": trial["phase"],
                        "trial_stage": trial["trial_stage"], "timestamp_ns": raw.timestamp_ns,
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
            elif (
                trial.get("trial_stage") != "start"
                and time.perf_counter_ns() - onset_ns >= int(float(self.exp["timeout_s"]) * 1e9)
            ):
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
        self._enable_intertrial_motion(trial)
        trial["events"].append({
            "event": "trial_end", "session_id": self.session_id,
            "trial_id": trial["trial_id"], "phase": trial["phase"],
            "trial_stage": trial["trial_stage"], "timestamp_ns": end_ns,
            "outcome": outcome, "valid": valid,
            "input_backend": self.input_backend,
        })
        await self._send({
            "type": "trial_result", "trial_id": trial["trial_id"],
            "trial_stage": trial["trial_stage"],
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
                "phase": trial["phase"], "trial_stage": trial["trial_stage"],
                "block_index": trial["block_index"],
                "cpi": trial["cpi"],
                "mousepad_condition": self.mousepad_condition,
                "input_backend": self.input_backend,
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
                "native_cursor_x_m": frame.get("native_cursor_x_m"),
                "native_cursor_y_m": frame.get("native_cursor_y_m"),
                "native_alignment_error_m": frame.get("native_alignment_error_m"),
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
        total_time_s = (trial["end_ns"] - trial["onset_ns"]) / 1e9
        cursor_displacement = float(np.linalg.norm(trial["cursor"] - trial["cursor_start"]))
        raw_trial_pos = trial["raw_trial_pos"]
        raw_session_start = trial["raw_session_start"]
        raw_session_pos = trial["raw_session_pos"]
        click_raw_trial_pos = trial.get("click_raw_trial_pos")
        click_raw_session_pos = trial.get("click_raw_session_pos")
        raw_displacement = float(np.linalg.norm(raw_trial_pos))
        raw_path_length = float(trial["raw_path_length"])
        raw_dt_values = np.asarray(trial.get("raw_dt_ns_values", []), dtype=np.float64)
        raw_event_rate = (
            len(trial["raw_rows"]) / total_time_s
            if total_time_s > 0
            else None
        )
        mean_polling_interval = (
            float(np.mean(raw_dt_values)) if raw_dt_values.size else None
        )
        std_polling_interval = (
            float(np.std(raw_dt_values)) if raw_dt_values.size else None
        )
        max_polling_gap = (
            float(np.max(raw_dt_values)) if raw_dt_values.size else None
        )
        cursor_points = np.asarray(
            [[trial["cursor_start"][0], trial["cursor_start"][1]]]
            + [
                [row["cursor_x_m"], row["cursor_y_m"]]
                for row in trial["raw_rows"]
            ],
            dtype=np.float64,
        )
        curvature_features = cursor_curvature_features(
            cursor_points,
            target_position=[template.target_x_m, template.target_y_m],
        )
        native_alignment_errors = np.asarray(
            [
                float(frame["native_alignment_error_m"])
                for frame in browser_frames
                if frame.get("native_alignment_error_m") is not None
                and math.isfinite(float(frame["native_alignment_error_m"]))
            ],
            dtype=np.float64,
        )
        native_raw_alignment_errors = np.asarray(
            trial.get("native_raw_alignment_errors", []), dtype=np.float64
        )
        summary = {
            "session_id": self.session_id, "trial_id": trial["trial_id"],
            "template_id": template.template_id, "phase": trial["phase"],
            "trial_stage": trial["trial_stage"],
            "block_index": trial["block_index"], "cpi": trial["cpi"],
            "cpi_setting_method": trial["cpi_setting_method"],
            "cpi_confirmed": trial["cpi_confirmed"],
            "mousepad_condition": self.mousepad_condition,
            "input_backend": self.input_backend,
            "cursor_presentation_mode": self.cursor_presentation_mode,
            "cursor_coordinate_source": (
                "windows_getcursorpos"
                if self.cursor_presentation_mode == "native_windows"
                else "raw_input_reconstruction"
            ),
            "attempt": trial["attempt"], "valid": trial["valid"],
            "invalid_reason": "" if trial["valid"] else trial["outcome"],
            "outcome": trial["outcome"], "success": trial["outcome"] == "success",
            "timeout": trial["outcome"] == "timeout", "onset_ns": trial["onset_ns"],
            "first_movement_ns": first_move, "click_ns": click_ns, "end_ns": trial["end_ns"],
            "reaction_time_s": None if first_move is None else (first_move - trial["onset_ns"]) / 1e9,
            "movement_time_s": None if first_move is None or click_ns is None else (click_ns - first_move) / 1e9,
            "total_time_s": total_time_s,
            "cursor_start_x_m": float(trial["cursor_start"][0]),
            "cursor_start_y_m": float(trial["cursor_start"][1]),
            "cursor_end_x_m": float(trial["cursor"][0]),
            "cursor_end_y_m": float(trial["cursor"][1]),
            "cursor_displacement_m": cursor_displacement,
            "target_x_m": template.target_x_m, "target_y_m": template.target_y_m,
            "target_radius_m": template.target_radius_m, "target_width_m": 2 * template.target_radius_m,
            "rendered_target_radius_px": client.get("rendered_target_radius_px"),
            "rendered_target_radius_m": rendered_radius_m, "rendered_fitts_id": rendered_id,
            "target_distance_m": template.target_distance_m, "fitts_id": template.fitts_id,
            "click_x_m": None if click_pos is None else float(click_pos[0]),
            "click_y_m": None if click_pos is None else float(click_pos[1]),
            "click_raw_trial_x_m": None if click_raw_trial_pos is None else float(click_raw_trial_pos[0]),
            "click_raw_trial_y_m": None if click_raw_trial_pos is None else float(click_raw_trial_pos[1]),
            "click_raw_session_x_m": None if click_raw_session_pos is None else float(click_raw_session_pos[0]),
            "click_raw_session_y_m": None if click_raw_session_pos is None else float(click_raw_session_pos[1]),
            "click_error_m": click_error, "minimum_target_distance_m": trial["minimum_distance"],
            "path_length_m": trial["path_length"],
            "straight_line_efficiency": None if trial["path_length"] == 0 else template.target_distance_m / trial["path_length"],
            "raw_event_count": len(trial["raw_rows"]), "frame_count": len(frame_rows),
            "late_frame_count": late_count, "max_frame_interval_ns": max_interval_ns,
            "focus_lost": not trial["valid"], "browser_onset_ms": trial["browser_onset_ms"],
            "raw_start_x_m": 0.0,
            "raw_start_y_m": 0.0,
            "raw_end_x_m": float(raw_trial_pos[0]),
            "raw_end_y_m": float(raw_trial_pos[1]),
            "raw_session_start_x_m": float(raw_session_start[0]),
            "raw_session_start_y_m": float(raw_session_start[1]),
            "raw_session_end_x_m": float(raw_session_pos[0]),
            "raw_session_end_y_m": float(raw_session_pos[1]),
            "raw_path_length_m": raw_path_length,
            "raw_displacement_m": raw_displacement,
            "raw_straight_line_efficiency": (
                None if raw_path_length == 0 else raw_displacement / raw_path_length
            ),
            "raw_mean_speed_m_s": (
                None if total_time_s <= 0 else raw_path_length / total_time_s
            ),
            "raw_max_speed_m_s": float(trial["raw_max_speed"]),
            "raw_event_rate_hz": raw_event_rate,
            "mean_polling_interval_ns": mean_polling_interval,
            "std_polling_interval_ns": std_polling_interval,
            "max_polling_gap_ns": max_polling_gap,
            "native_alignment_sample_count": int(native_alignment_errors.size),
            "native_alignment_mean_m": (
                float(np.mean(native_alignment_errors))
                if native_alignment_errors.size else None
            ),
            "native_alignment_p95_m": (
                float(np.percentile(native_alignment_errors, 95))
                if native_alignment_errors.size else None
            ),
            "native_alignment_max_m": (
                float(np.max(native_alignment_errors))
                if native_alignment_errors.size else None
            ),
            "native_raw_alignment_sample_count": int(native_raw_alignment_errors.size),
            "native_raw_alignment_mean_m": (
                float(np.mean(native_raw_alignment_errors))
                if native_raw_alignment_errors.size else None
            ),
            "native_raw_alignment_p95_m": (
                float(np.percentile(native_raw_alignment_errors, 95))
                if native_raw_alignment_errors.size else None
            ),
            "native_raw_alignment_max_m": (
                float(np.max(native_raw_alignment_errors))
                if native_raw_alignment_errors.size else None
            ),
            **curvature_features,
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
        if trial["trial_stage"] == "task":
            # ``_finish_trial`` initialized the inter-trial cursor at the click
            # endpoint; any movement received during saving has already moved it.
            next_cursor = self.current_cursor.copy()
            next_raw_session_pos = self.current_raw_session_pos.copy()
        else:
            next_cursor = trial["cursor"].copy()
            next_raw_session_pos = trial["raw_session_pos"].copy()
        if (
            trial["valid"]
            and trial["trial_stage"] == "start"
            and trial["outcome"] == "success"
        ):
            next_cursor = np.array(
                [template.target_x_m, template.target_y_m], dtype=np.float64
            )
            trial["events"].append({
                "event": "virtual_cursor_snap",
                "session_id": self.session_id,
                "trial_id": trial["trial_id"],
                "timestamp_ns": trial["end_ns"],
                "phase": trial["phase"],
                "trial_stage": trial["trial_stage"],
                "x_m": float(next_cursor[0]),
                "y_m": float(next_cursor[1]),
                "input_backend": self.input_backend,
            })
        assert self.logger is not None
        self.logger.submit_trial(summary, trial["raw_rows"], frame_rows, trial["events"])
        trial_id = trial["trial_id"]
        trial_stage = trial["trial_stage"]
        outcome = trial["outcome"]
        valid = trial["valid"]
        self.current_cursor = next_cursor
        self.current_raw_session_pos = next_raw_session_pos
        self.trial = None
        await self._send({
            "type": "trial_saved",
            "trial_id": trial_id,
            "trial_stage": trial_stage,
            "outcome": outcome,
            "valid": valid,
            "cursor_x_m": float(self.current_cursor[0]),
            "cursor_y_m": float(self.current_cursor[1]),
        })


def create_app(
    config: dict[str, Any],
    output_root: Path,
    input_backend: str = "auto",
    mousepad_condition: str = "unspecified",
) -> web.Application:
    @web.middleware
    async def no_cache_static(request: web.Request, handler):
        response = await handler(request)
        if request.path == "/" or request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    server = WebExperimentServer(
        config,
        output_root,
        input_backend=input_backend,
        mousepad_condition=mousepad_condition,
    )
    app = web.Application(middlewares=[no_cache_static])
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
        "--cursor-ab-pilot",
        action="store_true",
        help=(
            "Run a short cursor-presentation diagnostic with 3 practice and "
            "20 measured trials per CPI. The default full protocol is unchanged."
        ),
    )
    parser.add_argument(
        "--mousepad-condition",
        choices=("firm", "soft", "glass", "unspecified"),
        default="unspecified",
    )
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
    if args.cursor_ab_pilot:
        config["experiment"]["practice_trials_per_cpi"] = 3
        config["experiment"]["measured_trials_per_cpi"] = 20
        config["experiment"]["cursor_ab_pilot"] = True
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Point-and-click experiment: {url}")
    print(f"Input backend: {resolve_input_backend(args.input_backend)}")
    print(f"Mousepad condition: {args.mousepad_condition}")
    if args.cursor_ab_pilot:
        print("Cursor A/B pilot: 3 practice + 20 measured trials per CPI")
    print("Keep this window open during data collection. Press Ctrl+C to stop.")
    web.run_app(
        create_app(
            config,
            args.output_root,
            input_backend=args.input_backend,
            mousepad_condition=args.mousepad_condition,
        ),
        host=args.host,
        port=args.port,
        print=None,
    )


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
