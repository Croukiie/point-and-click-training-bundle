"""240 Hz human point-and-click study using Windows Raw Input.

Run from the repository root:
    python -m src.experiment.point_and_click_experiment --participant P001
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame
import yaml

from .async_logger import AsyncExperimentLogger
from .task_design import (
    angle_from_delta,
    counterbalanced_cpi_order,
    make_trial_stage_template,
    make_trial_templates,
    signed_angle_error,
    virtual_to_pixel,
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
    from .raw_input_win32 import RawInputMouse
except (ImportError, AttributeError):
    RawInputMouse = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
BACKEND_WINDOWS_RAW_INPUT = "windows_raw_input"
BACKEND_PYGAME_RELATIVE = "pygame_relative"

# Native 32 px frame extracted losslessly from the user-provided aero_arrow.cur.
# The CUR hotspot is (0, 0), matching the virtual cursor coordinates.
AERO_ARROW_CURSOR_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAA/klEQVR42u3VsWoCQRgE4Pl/"
    "XQgIt3dCTslbWN1T5BXzErlSLewCaYJw2kqKEyshJzFYjIWaiCTGwtsF2Sm3mY9hlwUkKtWk"
    "GbxFolI0HvhDSFSSpD/EHuAPcQTwgzgBuEf8AnCL+APgDnEGsP7aOECcAZDkcvlRM+IfAEnO"
    "54saERcASHI2e6doPLh2ffP0YLVao9W6AwBoIxk6fQVV9Ukgqp7z/uEV9LV5/+gEcChXk2Y"
    "idvw9u9hx7YDjcgAQTfLR6GW/QpLXusLuYv2UA4CaNIPYCUkWxZQi9s39J6lJXhTT3QoSvz"
    "oHNEz3AWInou0nNZ0eQkJCQkJCbjVbxl9Gb4rWTdMAAAAASUVORK5CYII="
)

WEB_THEME = {
    "bg": (11, 15, 23),
    "bg_2": (16, 22, 35),
    "panel": (18, 25, 38),
    "panel_soft": (255, 255, 255, 14),
    "line": (255, 255, 255, 28),
    "line_strong": (255, 255, 255, 46),
    "text": (244, 247, 251),
    "muted": (156, 168, 186),
    "muted_2": (118, 131, 153),
    "accent": (120, 215, 255),
    "accent_2": (154, 124, 255),
    "danger": (255, 123, 138),
}


class ExperimentAborted(Exception):
    pass


class ExperimentPaused(Exception):
    def __init__(self, trial_logged: bool = False):
        super().__init__("Experiment paused")
        self.trial_logged = trial_logged


@dataclass(frozen=True)
class TrialRunResult:
    valid: bool
    outcome: str
    cursor: np.ndarray
    trial_stage: str


class HighPrecisionPacer:
    """Hybrid sleep/spin pacing based on perf_counter_ns."""

    def __init__(self, frequency_hz: int, start_ns: int | None = None):
        self.period_ns = round(1_000_000_000 / frequency_hz)
        self.deadline_ns = time.perf_counter_ns() if start_ns is None else start_ns

    def wait_next(self):
        self.deadline_ns += self.period_ns
        while True:
            remaining_ns = self.deadline_ns - time.perf_counter_ns()
            if remaining_ns <= 0:
                break
            if remaining_ns > 900_000:
                time.sleep((remaining_ns - 500_000) / 1e9)
        now_ns = time.perf_counter_ns()
        if now_ns - self.deadline_ns > 4 * self.period_ns:
            self.deadline_ns = now_ns


@dataclass(frozen=True)
class RelativeMouseEvent:
    timestamp_ns: int
    device_handle: int
    device_name: str
    raw_flags: int
    button_flags: int
    button_data: int
    dx: int
    dy: int
    os_cursor_x_px: int | None
    os_cursor_y_px: int | None


class PygameRelativeMouse:
    """Cross-platform fallback using Pygame relative motion.

    This is not hardware Raw Input. It is intended for Mac/Linux pilots and UI
    testing, and is marked as ``pygame_relative`` in the logs.
    """

    def __init__(self):
        self._left_was_pressed = False

    def start(self):
        pygame.mouse.get_rel()
        self._left_was_pressed = bool(pygame.mouse.get_pressed()[0])

    def drain(self) -> list[RelativeMouseEvent]:
        dx, dy = pygame.mouse.get_rel()
        pressed = bool(pygame.mouse.get_pressed()[0])
        button_flags = RI_MOUSE_LEFT_BUTTON_DOWN if pressed and not self._left_was_pressed else 0
        self._left_was_pressed = pressed
        if dx == 0 and dy == 0 and button_flags == 0:
            return []
        return [
            RelativeMouseEvent(
                timestamp_ns=time.perf_counter_ns(),
                device_handle=0,
                device_name=BACKEND_PYGAME_RELATIVE,
                raw_flags=0,
                button_flags=button_flags,
                button_data=0,
                dx=int(dx),
                dy=int(dy),
                os_cursor_x_px=None,
                os_cursor_y_px=None,
            )
        ]

    def stop(self):
        return None


def resolve_native_input_backend(requested: str) -> str:
    if requested == "auto":
        return (
            BACKEND_WINDOWS_RAW_INPUT
            if sys.platform == "win32" and RawInputMouse is not None
            else BACKEND_PYGAME_RELATIVE
        )
    if requested == BACKEND_WINDOWS_RAW_INPUT and RawInputMouse is None:
        raise RuntimeError("Windows Raw Input backend is only available on Windows")
    return requested


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_dpi_aware() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass


def draw_centered_lines(surface, font, lines, color, vertical_offset=0):
    rendered = [font.render(line, True, color) for line in lines]
    total_height = sum(item.get_height() for item in rendered) + 12 * (len(rendered) - 1)
    y = (surface.get_height() - total_height) // 2 + vertical_offset
    for item in rendered:
        surface.blit(item, ((surface.get_width() - item.get_width()) // 2, y))
        y += item.get_height() + 12


def draw_wrapped_text(surface, text, font, color, rect, line_spacing=7, center=False):
    """Draw simple word-wrapped text and return the next y coordinate."""
    words = str(text).split()
    if not words:
        return rect.top
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= rect.width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)

    y = rect.top
    for line in lines:
        rendered = font.render(line, True, color)
        x = rect.centerx - rendered.get_width() // 2 if center else rect.left
        surface.blit(rendered, (x, y))
        y += rendered.get_height() + line_spacing
    return y


class PointAndClickExperiment:
    def __init__(self, args, config: dict[str, Any]):
        self.args = args
        self.config = config
        self.exp = config["experiment"]
        self.display_cfg = config["display"]
        self.appearance = config["appearance"]
        self.input_backend = resolve_native_input_backend(args.input_backend)
        self.mousepad_condition = args.mousepad_condition
        self.virtual_size = tuple(float(v) for v in self.display_cfg["virtual_size_m"])
        self.requested_resolution = tuple(int(v) for v in self.display_cfg["resolution_px"])
        self.refresh_hz = int(self.exp["refresh_hz"])
        self.nominal_frame_ns = round(1_000_000_000 / self.refresh_hz)
        self.mouse_gain = float(self.exp["mouse_gain"])
        self.mouse_reference_cpi = positive_cpi(
            self.exp.get("mouse_reference_cpi", DEFAULT_MOUSE_REFERENCE_CPI),
            name="experiment.mouse_reference_cpi",
        )
        self.cursor_meters_per_count = cursor_meters_per_count(
            self.mouse_gain,
            self.mouse_reference_cpi,
        )
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.participant}"
        self.trial_id = 0
        self.focused = True
        self.current_cursor = np.zeros(2, dtype=np.float64)
        self.current_raw_session_pos = np.zeros(2, dtype=np.float64)

        cpi_values = [int(v) for v in self.exp["cpi_values"]]
        self.cpi_order = counterbalanced_cpi_order(
            args.participant, cpi_values, int(self.exp["seed"])
        )
        template_count = int(self.exp["practice_trials_per_cpi"]) + int(
            self.exp["measured_trials_per_cpi"]
        )
        self.templates = make_trial_templates(
            template_count,
            self.virtual_size,
            tuple(float(v) for v in self.exp["target_id_range"]),
            int(self.exp["seed"]),
            cursor_position_m=None,
            target_radius_range_m=tuple(
                float(v)
                for v in self.exp.get("target_radius_range_m", (0.00125, 0.007))
            ),
        )

        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = ROOT / output_root
        self.output_dir = output_root / self.session_id
        self.screen = None
        self.font = None
        self.small_font = None
        self.title_font = None
        self.kicker_font = None
        self.button_font = None
        self.counter_font = None
        self.menu_background = None
        self.cursor_surface = None
        self.clock = None
        self.resolution = self.requested_resolution

    def session_metadata(self) -> dict[str, Any]:
        clock_info = time.get_clock_info("perf_counter")
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "input_backend": self.input_backend,
            "participant": {
                "id": self.args.participant,
                "age": self.args.age,
                "sex": self.args.sex,
                "dominant_hand": self.args.dominant_hand,
                "mouse_experience": self.args.mouse_experience,
                "gaming_hours_per_week": self.args.gaming_hours,
            },
            "hardware": {
                "mouse_model": self.args.mouse_model,
                "mousepad_condition": self.mousepad_condition,
                "declared_os_pointer_speed": self.args.os_pointer_speed,
                "declared_pointer_acceleration": self.args.pointer_acceleration,
                "display_profile_name": self.display_cfg.get("profile_name", "not_reported"),
                "display_resolution_px": list(self.resolution),
                "display_physical_size_m": self.display_cfg["physical_size_m"],
                "declared_refresh_hz": self.refresh_hz,
                "expected_mouse_polling_hz": self.config["logging"]["expected_mouse_polling_hz"],
            },
            "task": {
                "virtual_size_m": list(self.virtual_size),
                "mouse_gain": self.mouse_gain,
                "mouse_reference_cpi": self.mouse_reference_cpi,
                "cpi_order": self.cpi_order,
                "target_shape": "circle",
                "target_motion": "fixed",
                "first_click_ends_trial": True,
                "intertrial_cursor_motion": (
                    "Raw movement between trials sets the cursor start position "
                    "of the next trial and is excluded from the completed trial trajectory."
                ),
                "start_target_enabled": bool(self.exp.get("start_target_enabled", False)),
                "start_target_radius_m": float(self.exp.get("start_target_radius_m", 0.006)),
                "trial_stage_note": (
                    "Reference/start targets are disabled. Each target trial "
                    "starts from the current virtual cursor position after the "
                    "participant's separately handled inter-trial movement."
                    if not bool(self.exp.get("start_target_enabled", False))
                    else "Each logical trial is logged as a sampled start "
                    "target followed by its jointly sampled task target. After "
                    "a successful start click, the virtual cursor is snapped "
                    "to the start-target center so simulator geometry is exact."
                ),
                "target_onset_timestamp_note": (
                    "perf_counter_ns immediately after display.flip returns; not a photodiode measurement"
                ),
                "raw_timestamp_note": (
                    "perf_counter_ns when WM_INPUT is handled; the mouse does not provide a hardware timestamp"
                    if self.input_backend == BACKEND_WINDOWS_RAW_INPUT
                    else "perf_counter_ns when Pygame relative motion is sampled; not hardware Raw Input"
                ),
                "config": self.config,
            },
            "software": {
                "python": sys.version,
                "pygame": pygame.version.ver,
                "platform": platform.platform(),
                "perf_counter": {
                    "implementation": clock_info.implementation,
                    "monotonic": clock_info.monotonic,
                    "adjustable": clock_info.adjustable,
                    "resolution_s": clock_info.resolution,
                },
            },
            "notes": self.args.notes,
            "created_local": datetime.now().astimezone().isoformat(),
        }

    def initialize_display(self):
        set_dpi_aware()
        pygame.init()
        pygame.display.set_caption("Point-and-click CPI study")
        flags = pygame.DOUBLEBUF
        requested = self.requested_resolution
        if self.args.windowed:
            requested = (1280, 720)
            flags |= pygame.RESIZABLE
        else:
            flags |= pygame.FULLSCREEN
        try:
            self.screen = pygame.display.set_mode(requested, flags, vsync=1)
        except TypeError:
            self.screen = pygame.display.set_mode(requested, flags)
        if self.appearance.get("cursor_style") == "aero_arrow":
            cursor_bytes = base64.b64decode(AERO_ARROW_CURSOR_PNG_BASE64)
            self.cursor_surface = pygame.image.load(
                io.BytesIO(cursor_bytes), "aero_arrow.png"
            ).convert_alpha()
            cursor_size = tuple(
                int(value)
                for value in self.appearance.get("cursor_bitmap_size_px", (32, 32))
            )
            if self.cursor_surface.get_size() != cursor_size:
                self.cursor_surface = pygame.transform.smoothscale(
                    self.cursor_surface, cursor_size
                )
        self.resolution = self.screen.get_size()
        if not self.args.windowed and self.resolution != self.requested_resolution:
            raise RuntimeError(
                f"Expected {self.requested_resolution}, got {self.resolution}. "
                "Use the fixed 24 inch Full HD display profile before collecting data."
            )
        pygame.mouse.set_visible(False)
        # Even in debug/windowed mode the OS cursor must be captured. Otherwise
        # it can leave the Pygame window and clicks may go to another window
        # instead of being handled reliably by the Raw Input experiment loop.
        pygame.event.set_grab(True)
        if self.input_backend == BACKEND_WINDOWS_RAW_INPUT:
            pygame.event.set_blocked(
                [pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL]
            )
        self.title_font = pygame.font.SysFont("segoeui", 58, bold=True)
        self.font = pygame.font.SysFont("segoeui", 31)
        self.small_font = pygame.font.SysFont("segoeui", 21)
        self.kicker_font = pygame.font.SysFont("segoeui", 15, bold=True)
        self.button_font = pygame.font.SysFont("segoeui", 20, bold=True)
        self.counter_font = pygame.font.SysFont(
            "segoeui", int(self.appearance.get("phase_counter_font_px", 14))
        )
        self.clock = pygame.time.Clock()
        self._refresh_menu_background()

    def _refresh_menu_background(self):
        width, height = self.resolution
        background = pygame.Surface((width, height)).convert()
        background.fill(WEB_THEME["bg"])

        # Subtle web-like glow and grid, cached once so it does not touch the
        # 240 Hz trial loop.
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        glow_specs = [
            ((round(width * 0.12), round(height * 0.13)), round(width * 0.22), (*WEB_THEME["accent"], 30)),
            ((round(width * 0.84), round(height * 0.18)), round(width * 0.20), (*WEB_THEME["accent_2"], 24)),
            ((round(width * 0.73), round(height * 0.88)), round(width * 0.24), (*self.appearance["target_rgb"], 20)),
        ]
        for center, radius, color in glow_specs:
            for step in range(9, 0, -1):
                alpha = max(0, color[3] * step // 15)
                pygame.draw.circle(overlay, (*color[:3], alpha), center, radius * step // 9)

        grid_color = (255, 255, 255, 13)
        for x in range(0, width, 44):
            pygame.draw.line(overlay, grid_color, (x, 0), (x, height))
        for y in range(0, height, 44):
            pygame.draw.line(overlay, grid_color, (0, y), (width, y))

        background.blit(overlay, (0, 0))
        self.menu_background = background

    def _fill_menu_background(self):
        if self.menu_background is None or self.menu_background.get_size() != self.screen.get_size():
            self.resolution = self.screen.get_size()
            self._refresh_menu_background()
        self.screen.blit(self.menu_background, (0, 0))

    def _fill_task_background(self):
        self.screen.fill(self.appearance["background_rgb"])

    def _draw_pill(self, text, center, color=None, text_color=None, padding=(18, 10)):
        color = color or (7, 11, 18, 150)
        text_color = text_color or WEB_THEME["muted"]
        rendered = self.button_font.render(text, True, text_color)
        rect = pygame.Rect(0, 0, rendered.get_width() + padding[0] * 2, rendered.get_height() + padding[1] * 2)
        rect.center = center
        pill = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(pill, color, pill.get_rect(), border_radius=rect.height // 2)
        pygame.draw.rect(pill, WEB_THEME["line_strong"], pill.get_rect(), width=1, border_radius=rect.height // 2)
        pill.blit(rendered, (padding[0], padding[1]))
        self.screen.blit(pill, rect)
        return rect

    def _draw_card_screen(
        self,
        title: str,
        body_lines: list[str],
        *,
        kicker: str = "Point-and-click CPI study",
        primary_action: str | None = "Press SPACE to continue",
        secondary_actions: list[str] | None = None,
        danger_action: str | None = None,
    ):
        self._fill_menu_background()
        width, height = self.screen.get_size()
        card_width = min(760, width - 80)
        card_height = min(520, height - 80)
        card = pygame.Rect(0, 0, card_width, card_height)
        card.center = (width // 2, height // 2)

        shadow = pygame.Surface((card.width + 80, card.height + 80), pygame.SRCALPHA)
        pygame.draw.rect(
            shadow,
            (0, 0, 0, 85),
            shadow.get_rect().inflate(-40, -40),
            border_radius=34,
        )
        self.screen.blit(shadow, (card.left - 40, card.top - 32))

        panel = pygame.Surface(card.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (*WEB_THEME["panel"], 238), panel.get_rect(), border_radius=26)
        pygame.draw.rect(panel, WEB_THEME["line"], panel.get_rect(), width=1, border_radius=26)
        pygame.draw.rect(panel, (*WEB_THEME["panel_soft"][:3], WEB_THEME["panel_soft"][3]), panel.get_rect().inflate(-2, -2), width=1, border_radius=25)
        self.screen.blit(panel, card)

        content = card.inflate(-84, -76)
        y = content.top
        kicker_surface = self.kicker_font.render(kicker.upper(), True, WEB_THEME["accent"])
        self.screen.blit(kicker_surface, (content.centerx - kicker_surface.get_width() // 2, y))
        y += kicker_surface.get_height() + 18

        title_surface = self.title_font.render(title, True, WEB_THEME["text"])
        self.screen.blit(title_surface, (content.centerx - title_surface.get_width() // 2, y))
        y += title_surface.get_height() + 28

        for line in body_lines:
            y = draw_wrapped_text(
                self.screen,
                line,
                self.small_font,
                WEB_THEME["muted"],
                pygame.Rect(content.left, y, content.width, 80),
                center=True,
            )
            y += 8

        actions = []
        if primary_action:
            actions.append((primary_action, WEB_THEME["accent"], (120, 215, 255, 30)))
        for action in secondary_actions or []:
            actions.append((action, WEB_THEME["muted"], (255, 255, 255, 18)))
        if danger_action:
            actions.append((danger_action, (255, 216, 222), (255, 123, 138, 34)))

        if actions:
            action_y = min(card.bottom - 78, max(y + 14, card.centery + 120))
            total_width = sum(self.button_font.size(label)[0] + 42 for label, *_ in actions) + 12 * (len(actions) - 1)
            x = content.centerx - total_width // 2
            for label, text_color, bg_color in actions:
                label_width = self.button_font.size(label)[0] + 42
                self._draw_pill(
                    label,
                    (x + label_width // 2, action_y),
                    color=(*bg_color[:3], bg_color[3]),
                    text_color=text_color,
                    padding=(21, 10),
                )
                x += label_width + 12

    def _handle_control_events(self, allow_pause: bool = True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise ExperimentAborted()
            if (
                allow_pause
                and event.type == pygame.KEYDOWN
                and event.key in {pygame.K_ESCAPE, pygame.K_p}
            ):
                raise ExperimentPaused()
            if event.type == pygame.WINDOWFOCUSLOST:
                self.focused = False
            elif event.type == pygame.WINDOWFOCUSGAINED:
                self.focused = True
            yield event

    def main_instruction_lines(self) -> list[str]:
        geometry_lines = (
            [
                "Each task begins with a red start target.",
                "Click red to set the start, then point to the light-orange task target.",
            ]
            if bool(self.exp.get("start_target_enabled", False))
            else [
                "There is no center reference target; you may move freely between trials."
            ]
        )
        return [
            "Point-and-click experiment",
            "The random target is light orange; the cursor is the mouse arrow.",
            *geometry_lines,
            "The first click ends each trial, even when it misses.",
            "Press P or Escape to open the pause menu.",
            "Press SPACE to continue.",
        ]

    def instruction_screen(self, lines: list[str], wait_for_space=True, duration_s=None):
        start = time.perf_counter()
        title = lines[0] if lines else "Point-and-click experiment"
        body = lines[1:] if len(lines) > 1 else []
        while True:
            for event in self._handle_control_events(allow_pause=False):
                if wait_for_space and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return
            if duration_s is not None and time.perf_counter() - start >= duration_s:
                return
            self._draw_card_screen(
                title,
                body,
                primary_action="Press SPACE to continue" if wait_for_space else None,
            )
            pygame.display.flip()
            self.clock.tick(60)

    def pause_menu(self, logger: AsyncExperimentLogger, raw_mouse: RawInputMouse):
        logger.submit_event({
            "event": "pause_start", "session_id": self.session_id,
            "timestamp_ns": time.perf_counter_ns(),
        })
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise ExperimentAborted()
                    if event.type == pygame.WINDOWFOCUSLOST:
                        self.focused = False
                    elif event.type == pygame.WINDOWFOCUSGAINED:
                        self.focused = True
                    if event.type != pygame.KEYDOWN:
                        continue
                    if event.key == pygame.K_SPACE:
                        logger.submit_event({
                            "event": "pause_end", "session_id": self.session_id,
                            "timestamp_ns": time.perf_counter_ns(),
                        })
                        return
                    if event.key == pygame.K_m:
                        self.instruction_screen(self.main_instruction_lines())
                    if event.key == pygame.K_q:
                        raise ExperimentAborted()
                raw_mouse.drain()
                self._draw_card_screen(
                    "Experiment paused",
                    [
                        "The current trial has been safely paused and can be repeated if needed.",
                        "You can resume the experiment, review the main instructions, or save and stop.",
                    ],
                    kicker="Main menu",
                    primary_action="SPACE: resume",
                    secondary_actions=["M: instructions"],
                    danger_action="Q: save and quit",
                )
                pygame.display.flip()
                self.clock.tick(60)
        finally:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            raw_mouse.drain()

    def _draw_cursor(self, cursor: np.ndarray):
        cursor_px = virtual_to_pixel(
            float(cursor[0]), float(cursor[1]), self.virtual_size, self.resolution
        )
        x, y = int(cursor_px[0]), int(cursor_px[1])

        if (
            self.appearance.get("cursor_style") == "aero_arrow"
            and self.cursor_surface is not None
        ):
            self.screen.blit(self.cursor_surface, (x, y))
            return

        if self.appearance.get("cursor_style", "arrow") == "arrow":
            width = max(6, int(self.appearance.get("cursor_arrow_width_px", 12)))
            height = max(10, int(self.appearance.get("cursor_arrow_height_px", 18)))
            outline_width = max(
                1, int(self.appearance.get("cursor_arrow_outline_px", 1))
            )
            # Compact classic Windows cursor. ``(x, y)`` remains its hotspot
            # independently of the visible polygon.
            points = [
                (x, y),
                (x, round(y + height * 0.78)),
                (round(x + width * 0.28), round(y + height * 0.62)),
                (round(x + width * 0.53), y + height),
                (round(x + width * 0.75), round(y + height * 0.91)),
                (round(x + width * 0.48), round(y + height * 0.58)),
                (x + width, round(y + height * 0.58)),
            ]
            shadow_offset = max(
                0, int(self.appearance.get("cursor_shadow_offset_px", 1))
            )
            if shadow_offset:
                shadow_points = [
                    (px + shadow_offset, py + shadow_offset) for px, py in points
                ]
                pygame.draw.polygon(
                    self.screen,
                    self.appearance.get("cursor_shadow_rgb", (0, 0, 0)),
                    shadow_points,
                )
            pygame.draw.polygon(self.screen, self.appearance["cursor_rgb"], points)
            pygame.draw.polygon(
                self.screen,
                self.appearance.get("cursor_outline_rgb", (32, 32, 32)),
                points,
                width=outline_width,
            )
            return

        gap = int(self.appearance.get("cursor_crosshair_gap_px", 18))
        length = int(self.appearance.get("cursor_crosshair_length_px", 60))
        thickness = max(1, int(self.appearance.get("cursor_line_width_px", 8)))
        center_size = max(1, int(self.appearance.get("cursor_crosshair_center_px", thickness)))
        rects = [
            pygame.Rect(x - gap - length, y - thickness // 2, length, thickness),
            pygame.Rect(x + gap, y - thickness // 2, length, thickness),
            pygame.Rect(x - thickness // 2, y - gap - length, thickness, length),
            pygame.Rect(x - thickness // 2, y + gap, thickness, length),
            pygame.Rect(x - center_size // 2, y - center_size // 2, center_size, center_size),
        ]
        for rect in rects:
            pygame.draw.rect(self.screen, self.appearance["cursor_rgb"], rect)

    def draw_inter_trial_cursor(self, cursor: np.ndarray):
        self._fill_task_background()
        self._draw_cursor(cursor)

    def _draw_phase_trial_counter(self, current: int, total: int):
        if self.counter_font is None or total <= 0:
            return
        rendered = self.counter_font.render(
            f"{current} / {total}",
            True,
            self.appearance.get("phase_counter_rgb", (230, 232, 236)),
        )
        rendered.set_alpha(
            round(255 * float(self.appearance.get("phase_counter_alpha", 0.48)))
        )
        margin = int(self.appearance.get("phase_counter_margin_px", 18))
        self.screen.blit(rendered, (self.resolution[0] - margin - rendered.get_width(), margin))

    def draw_trial(
        self,
        cursor: np.ndarray,
        template,
        color=None,
        trial_stage: str = "task",
        phase_trial_index: int | None = None,
        phase_trial_total: int | None = None,
    ):
        self._fill_task_background()
        target_px = virtual_to_pixel(
            template.target_x_m, template.target_y_m, self.virtual_size, self.resolution
        )
        px_per_m = self.resolution[0] / self.virtual_size[0]
        target_radius_px = max(1, round(template.target_radius_m * px_per_m))
        target_color = color or (
            self.appearance.get("start_target_rgb", (235, 64, 52))
            if trial_stage == "start"
            else self.appearance["target_rgb"]
        )
        pygame.draw.circle(
            self.screen,
            target_color,
            target_px,
            target_radius_px,
        )
        if phase_trial_index is not None and phase_trial_total is not None:
            self._draw_phase_trial_counter(phase_trial_index, phase_trial_total)
        self._draw_cursor(cursor)

    def rendered_target_geometry(self, template) -> tuple[int, float, float]:
        px_per_m = self.resolution[0] / self.virtual_size[0]
        radius_px = max(1, round(template.target_radius_m * px_per_m))
        radius_m = radius_px / px_per_m
        rendered_id = math.log2(template.target_distance_m / (2.0 * radius_m) + 1.0)
        return radius_px, radius_m, rendered_id

    def _blank_interval(
        self,
        duration_s: float,
        raw_mouse: RawInputMouse,
        cursor: np.ndarray | None = None,
        *,
        cpi: int | None = None,
        allow_cursor_motion: bool = False,
    ) -> np.ndarray | None:
        visible_cursor = None if cursor is None else cursor.copy()
        raw_m_per_count = physical_meters_per_count(cpi) if cpi is not None else None
        start = time.perf_counter()
        while time.perf_counter() - start < duration_s:
            for _ in self._handle_control_events():
                pass
            consumed = raw_mouse.drain()
            if allow_cursor_motion and visible_cursor is not None:
                for raw in consumed:
                    visible_cursor[0] += raw.dx * self.cursor_meters_per_count
                    visible_cursor[1] -= raw.dy * self.cursor_meters_per_count
                    visible_cursor[0] = np.clip(
                        visible_cursor[0], -self.virtual_size[0] / 2, self.virtual_size[0] / 2
                    )
                    visible_cursor[1] = np.clip(
                        visible_cursor[1], -self.virtual_size[1] / 2, self.virtual_size[1] / 2
                    )
                    if raw_m_per_count is not None:
                        self.current_raw_session_pos += np.array(
                            [raw.dx * raw_m_per_count, -raw.dy * raw_m_per_count],
                            dtype=np.float64,
                        )
            if visible_cursor is None:
                self._fill_task_background()
            else:
                self.draw_inter_trial_cursor(visible_cursor)
            pygame.display.flip()
            self.clock.tick(120)
        return visible_cursor

    def run_trial(
        self,
        base_template,
        cpi: int,
        block_index: int,
        phase: str,
        attempt: int,
        phase_trial_index: int,
        phase_trial_total: int,
        raw_mouse: RawInputMouse,
        logger: AsyncExperimentLogger,
        trial_stage: str = "task",
    ) -> TrialRunResult:
        start_enabled = bool(self.exp.get("start_target_enabled", False))
        intertrial_cursor = self._blank_interval(
            float(self.exp["inter_trial_s"]),
            raw_mouse,
            cursor=self.current_cursor.copy(),
            cpi=cpi,
            allow_cursor_motion=(not start_enabled or trial_stage == "start"),
        )
        if intertrial_cursor is not None:
            self.current_cursor = intertrial_cursor
        self.trial_id += 1
        trial_id = self.trial_id
        if not start_enabled:
            trial_stage = "task"
        cursor = self.current_cursor.copy()
        template = make_trial_stage_template(
            base_template,
            trial_stage,
            cursor,
            self.virtual_size,
            float(self.exp.get("start_target_radius_m", 0.006)),
            tuple(float(v) for v in self.exp["target_radius_range_m"]),
            preserve_sampled_task_geometry=(
                self.exp.get("protocol_version") == "simulator_geometry_exact_v3"
            ),
        )
        cursor_start = cursor.copy()
        raw_mouse.drain()

        self.draw_trial(
            cursor,
            template,
            trial_stage=trial_stage,
            phase_trial_index=phase_trial_index,
            phase_trial_total=phase_trial_total,
        )
        pygame.display.flip()
        onset_ns = time.perf_counter_ns()
        events_log = [{
            "event": "target_onset", "session_id": self.session_id,
            "trial_id": trial_id, "timestamp_ns": onset_ns,
            "phase": phase, "trial_stage": trial_stage,
            "input_backend": self.input_backend,
        }]

        raw_rows: list[dict] = []
        frame_rows: list[dict] = []
        event_index = 0
        frame_index = 0
        first_movement_ns = None
        click_ns = None
        click_pos = None
        outcome = None
        end_ns = None
        path_length = 0.0
        minimum_distance = float(np.linalg.norm(cursor - [template.target_x_m, template.target_y_m]))
        previous_raw_ns = None
        previous_present_ns = onset_ns
        late_frame_count = 0
        focus_lost = False
        participant_paused = False
        max_frame_interval_ns = 0
        scale_m_per_count = self.cursor_meters_per_count
        raw_m_per_count = physical_meters_per_count(cpi)
        raw_trial_pos = np.zeros(2, dtype=np.float64)
        raw_session_start = self.current_raw_session_pos.copy()
        raw_session_pos = self.current_raw_session_pos.copy()
        raw_path_length = 0.0
        raw_max_speed = 0.0
        raw_dt_ns_values: list[int] = []
        previous_cursor_velocity = None
        previous_raw_velocity = None
        click_raw_trial_pos = None
        click_raw_session_pos = None
        pacer = HighPrecisionPacer(self.refresh_hz, onset_ns)

        while outcome is None:
            pacer.wait_next()
            frame_start_ns = time.perf_counter_ns()
            try:
                for _ in self._handle_control_events():
                    pass
            except ExperimentPaused:
                participant_paused = True
                outcome = "participant_paused"
                end_ns = time.perf_counter_ns()
                events_log.append({
                    "event": "participant_pause_requested",
                    "session_id": self.session_id,
                    "trial_id": trial_id,
                    "timestamp_ns": end_ns,
                    "input_backend": self.input_backend,
                })
            if not self.focused:
                focus_lost = True
                outcome = "focus_lost"
                end_ns = time.perf_counter_ns()

            consumed = raw_mouse.drain()
            for raw in consumed:
                if raw.timestamp_ns < onset_ns:
                    continue
                old_cursor = cursor.copy()
                cursor[0] += raw.dx * scale_m_per_count
                cursor[1] -= raw.dy * scale_m_per_count
                cursor[0] = np.clip(cursor[0], -self.virtual_size[0] / 2, self.virtual_size[0] / 2)
                cursor[1] = np.clip(cursor[1], -self.virtual_size[1] / 2, self.virtual_size[1] / 2)
                movement = cursor - old_cursor
                movement_norm = float(np.linalg.norm(movement))
                path_length += movement_norm
                if movement_norm > 0 and first_movement_ns is None:
                    first_movement_ns = raw.timestamp_ns
                dt_ns = 0 if previous_raw_ns is None else max(0, raw.timestamp_ns - previous_raw_ns)
                if previous_raw_ns is not None:
                    raw_dt_ns_values.append(dt_ns)
                dt_s = dt_ns / 1e9 if dt_ns else 0.0
                if dt_ns > 0:
                    velocity = movement / dt_s
                else:
                    velocity = np.zeros(2)
                cursor_acceleration = (
                    (velocity - previous_cursor_velocity) / dt_s
                    if previous_cursor_velocity is not None and dt_s > 0
                    else np.zeros(2)
                )
                raw_delta = np.array(
                    [float(raw.dx) * raw_m_per_count, -float(raw.dy) * raw_m_per_count],
                    dtype=np.float64,
                )
                raw_trial_pos = raw_trial_pos + raw_delta
                raw_session_pos = raw_session_pos + raw_delta
                raw_movement_norm = float(np.linalg.norm(raw_delta))
                raw_path_length += raw_movement_norm
                raw_velocity = raw_delta / dt_s if dt_s > 0 else np.zeros(2)
                raw_acceleration = (
                    (raw_velocity - previous_raw_velocity) / dt_s
                    if previous_raw_velocity is not None and dt_s > 0
                    else np.zeros(2)
                )
                raw_speed = float(np.linalg.norm(raw_velocity))
                raw_max_speed = max(raw_max_speed, raw_speed)
                distance = float(
                    np.linalg.norm(cursor - [template.target_x_m, template.target_y_m])
                )
                minimum_distance = min(minimum_distance, distance)
                cursor_heading = angle_from_delta(movement)
                target_vector = np.array([template.target_x_m, template.target_y_m]) - cursor
                target_angle = angle_from_delta(target_vector)
                raw_angle = angle_from_delta(raw_delta)
                heading_error = signed_angle_error(cursor_heading, target_angle)
                inside = distance <= template.target_radius_m
                event_index += 1
                raw_rows.append({
                    "session_id": self.session_id, "trial_id": trial_id, "phase": phase,
                    "trial_stage": trial_stage,
                    "block_index": block_index, "cpi": cpi,
                    "mousepad_condition": self.mousepad_condition,
                    "input_backend": self.input_backend,
                    "event_index": event_index,
                    "timestamp_ns": raw.timestamp_ns,
                    "time_from_onset_ns": raw.timestamp_ns - onset_ns,
                    "device_handle": raw.device_handle,
                    "device_name": raw.device_name,
                    "os_cursor_x_px": raw.os_cursor_x_px,
                    "os_cursor_y_px": raw.os_cursor_y_px,
                    "browser_timestamp_ms": None,
                    "raw_flags": raw.raw_flags,
                    "button_flags": raw.button_flags, "button_data": raw.button_data,
                    "dx_counts": raw.dx, "dy_counts": raw.dy, "dt_ns": dt_ns,
                    "raw_dx_m": float(raw_delta[0]), "raw_dy_m": float(raw_delta[1]),
                    "raw_trial_x_m": float(raw_trial_pos[0]),
                    "raw_trial_y_m": float(raw_trial_pos[1]),
                    "raw_session_x_m": float(raw_session_pos[0]),
                    "raw_session_y_m": float(raw_session_pos[1]),
                    "raw_path_length_m": float(raw_path_length),
                    "raw_displacement_from_trial_start_m": float(np.linalg.norm(raw_trial_pos)),
                    "raw_speed_m_s": raw_speed,
                    "raw_angle_rad": raw_angle,
                    "raw_acceleration_x_m_s2": float(raw_acceleration[0]),
                    "raw_acceleration_y_m_s2": float(raw_acceleration[1]),
                    "raw_acceleration_m_s2": float(np.linalg.norm(raw_acceleration)),
                    "cursor_delta_x_m": float(movement[0]),
                    "cursor_delta_y_m": float(movement[1]),
                    "cursor_heading_rad": cursor_heading,
                    "cursor_path_length_m": float(path_length),
                    "cursor_x_m": float(cursor[0]), "cursor_y_m": float(cursor[1]),
                    "velocity_x_m_s": float(velocity[0]), "velocity_y_m_s": float(velocity[1]),
                    "speed_m_s": float(np.linalg.norm(velocity)),
                    "cursor_acceleration_x_m_s2": float(cursor_acceleration[0]),
                    "cursor_acceleration_y_m_s2": float(cursor_acceleration[1]),
                    "cursor_acceleration_m_s2": float(np.linalg.norm(cursor_acceleration)),
                    "target_x_m": template.target_x_m, "target_y_m": template.target_y_m,
                    "target_radius_m": template.target_radius_m,
                    "distance_to_target_m": distance,
                    "angle_to_target_rad": target_angle,
                    "heading_error_rad": heading_error,
                    "inside_target": inside,
                })
                previous_raw_ns = raw.timestamp_ns
                previous_cursor_velocity = velocity.copy()
                previous_raw_velocity = raw_velocity.copy()

                if outcome is None and raw.button_flags & RI_MOUSE_LEFT_BUTTON_DOWN:
                    click_ns = raw.timestamp_ns
                    click_pos = cursor.copy()
                    click_raw_trial_pos = raw_trial_pos.copy()
                    click_raw_session_pos = raw_session_pos.copy()
                    outcome = "success" if inside else "miss"
                    end_ns = click_ns
                    events_log.append({
                        "event": "click", "session_id": self.session_id,
                        "trial_id": trial_id, "timestamp_ns": click_ns,
                        "phase": phase, "trial_stage": trial_stage,
                        "x_m": float(cursor[0]), "y_m": float(cursor[1]), "success": inside,
                        "input_backend": self.input_backend,
                    })
                    # A click terminates the trial. Do not accidentally process
                    # later movement events already present in this drain batch.
                    break

            now_ns = time.perf_counter_ns()
            if (
                outcome is None
                and trial_stage != "start"
                and now_ns - onset_ns >= int(float(self.exp["timeout_s"]) * 1e9)
            ):
                outcome = "timeout"
                end_ns = now_ns

            self.draw_trial(
                cursor,
                template,
                trial_stage=trial_stage,
                phase_trial_index=phase_trial_index,
                phase_trial_total=phase_trial_total,
            )
            pygame.display.flip()
            present_ns = time.perf_counter_ns()
            frame_interval_ns = present_ns - previous_present_ns
            max_frame_interval_ns = max(max_frame_interval_ns, frame_interval_ns)
            late_by_ns = max(0, frame_interval_ns - self.nominal_frame_ns)
            if frame_interval_ns > self.nominal_frame_ns * 1.5:
                late_frame_count += 1
            distance = float(np.linalg.norm(cursor - [template.target_x_m, template.target_y_m]))
            frame_index += 1
            frame_rows.append({
                "session_id": self.session_id, "trial_id": trial_id, "phase": phase,
                "trial_stage": trial_stage,
                "block_index": block_index, "cpi": cpi,
                "mousepad_condition": self.mousepad_condition,
                "input_backend": self.input_backend,
                "frame_index": frame_index,
                "frame_start_ns": frame_start_ns, "present_return_ns": present_ns,
                "time_from_onset_ns": present_ns - onset_ns,
                "frame_interval_ns": frame_interval_ns, "late_by_ns": late_by_ns,
                "raw_events_consumed": len(consumed), "cursor_x_m": float(cursor[0]),
                "cursor_y_m": float(cursor[1]), "target_x_m": template.target_x_m,
                "target_y_m": template.target_y_m, "target_radius_m": template.target_radius_m,
                "distance_to_target_m": distance,
                "inside_target": distance <= template.target_radius_m,
                "window_focused": self.focused,
            })
            previous_present_ns = present_ns

        assert end_ns is not None
        rendered_radius_px, rendered_radius_m, rendered_id = self.rendered_target_geometry(template)
        valid = not focus_lost and not participant_paused
        click_error = None if click_pos is None else float(
            np.linalg.norm(click_pos - [template.target_x_m, template.target_y_m])
        )
        invalid_reason = ""
        if focus_lost:
            invalid_reason = "focus_lost"
        elif participant_paused:
            invalid_reason = "participant_paused"
        total_time_s = (end_ns - onset_ns) / 1e9
        cursor_displacement = float(np.linalg.norm(cursor - cursor_start))
        raw_displacement = float(np.linalg.norm(raw_trial_pos))
        raw_dt_values = np.asarray(raw_dt_ns_values, dtype=np.float64)
        raw_event_rate = len(raw_rows) / total_time_s if total_time_s > 0 else None
        mean_polling_interval = float(np.mean(raw_dt_values)) if raw_dt_values.size else None
        std_polling_interval = float(np.std(raw_dt_values)) if raw_dt_values.size else None
        max_polling_gap = float(np.max(raw_dt_values)) if raw_dt_values.size else None
        cursor_points = np.asarray(
            [[cursor_start[0], cursor_start[1]]]
            + [[row["cursor_x_m"], row["cursor_y_m"]] for row in raw_rows],
            dtype=np.float64,
        )
        curvature_features = cursor_curvature_features(
            cursor_points,
            target_position=[template.target_x_m, template.target_y_m],
        )
        summary = {
            "session_id": self.session_id, "trial_id": trial_id,
            "template_id": template.template_id, "phase": phase,
            "trial_stage": trial_stage,
            "block_index": block_index, "cpi": cpi,
            "mousepad_condition": self.mousepad_condition,
            "input_backend": self.input_backend,
            "attempt": attempt,
            "valid": valid, "invalid_reason": invalid_reason,
            "outcome": outcome, "success": outcome == "success", "timeout": outcome == "timeout",
            "onset_ns": onset_ns, "first_movement_ns": first_movement_ns,
            "click_ns": click_ns, "end_ns": end_ns,
            "reaction_time_s": None if first_movement_ns is None else (first_movement_ns - onset_ns) / 1e9,
            "movement_time_s": None if click_ns is None or first_movement_ns is None else (click_ns - first_movement_ns) / 1e9,
            "total_time_s": total_time_s,
            "cursor_start_x_m": float(cursor_start[0]), "cursor_start_y_m": float(cursor_start[1]),
            "cursor_end_x_m": float(cursor[0]), "cursor_end_y_m": float(cursor[1]),
            "cursor_displacement_m": cursor_displacement,
            "target_x_m": template.target_x_m, "target_y_m": template.target_y_m,
            "target_radius_m": template.target_radius_m,
            "target_width_m": 2.0 * template.target_radius_m,
            "rendered_target_radius_px": rendered_radius_px,
            "rendered_target_radius_m": rendered_radius_m,
            "rendered_fitts_id": rendered_id,
            "target_distance_m": template.target_distance_m, "fitts_id": template.fitts_id,
            "click_x_m": None if click_pos is None else float(click_pos[0]),
            "click_y_m": None if click_pos is None else float(click_pos[1]),
            "click_raw_trial_x_m": None if click_raw_trial_pos is None else float(click_raw_trial_pos[0]),
            "click_raw_trial_y_m": None if click_raw_trial_pos is None else float(click_raw_trial_pos[1]),
            "click_raw_session_x_m": None if click_raw_session_pos is None else float(click_raw_session_pos[0]),
            "click_raw_session_y_m": None if click_raw_session_pos is None else float(click_raw_session_pos[1]),
            "click_error_m": click_error, "minimum_target_distance_m": minimum_distance,
            "path_length_m": path_length,
            "straight_line_efficiency": None if path_length == 0 else template.target_distance_m / path_length,
            "raw_event_count": len(raw_rows), "frame_count": len(frame_rows),
            "late_frame_count": late_frame_count,
            "max_frame_interval_ns": max_frame_interval_ns, "focus_lost": focus_lost,
            "raw_start_x_m": 0.0,
            "raw_start_y_m": 0.0,
            "raw_end_x_m": float(raw_trial_pos[0]),
            "raw_end_y_m": float(raw_trial_pos[1]),
            "raw_session_start_x_m": float(raw_session_start[0]),
            "raw_session_start_y_m": float(raw_session_start[1]),
            "raw_session_end_x_m": float(raw_session_pos[0]),
            "raw_session_end_y_m": float(raw_session_pos[1]),
            "raw_path_length_m": float(raw_path_length),
            "raw_displacement_m": raw_displacement,
            "raw_straight_line_efficiency": (
                None if raw_path_length == 0 else raw_displacement / raw_path_length
            ),
            "raw_mean_speed_m_s": None if total_time_s <= 0 else raw_path_length / total_time_s,
            "raw_max_speed_m_s": raw_max_speed,
            "raw_event_rate_hz": raw_event_rate,
            "mean_polling_interval_ns": mean_polling_interval,
            "std_polling_interval_ns": std_polling_interval,
            "max_polling_gap_ns": max_polling_gap,
            **curvature_features,
        }
        events_log.append({
            "event": "trial_end", "session_id": self.session_id, "trial_id": trial_id,
            "timestamp_ns": end_ns, "outcome": outcome, "valid": valid,
            "phase": phase, "trial_stage": trial_stage,
            "input_backend": self.input_backend,
        })
        next_cursor = cursor.copy()
        if valid and trial_stage == "start" and outcome == "success":
            next_cursor = np.array(
                [template.target_x_m, template.target_y_m], dtype=np.float64
            )
            events_log.append({
                "event": "virtual_cursor_snap",
                "session_id": self.session_id,
                "trial_id": trial_id,
                "timestamp_ns": end_ns,
                "phase": phase,
                "trial_stage": trial_stage,
                "x_m": float(next_cursor[0]),
                "y_m": float(next_cursor[1]),
                "input_backend": self.input_backend,
            })
        logger.submit_trial(summary, raw_rows, frame_rows, events_log)
        self.current_cursor = next_cursor
        self.current_raw_session_pos = raw_session_pos.copy()

        if participant_paused:
            raise ExperimentPaused(trial_logged=True)

        skip_feedback = trial_stage == "start" and outcome == "success"
        if valid and not skip_feedback:
            self.draw_inter_trial_cursor(cursor)
            pygame.display.flip()
            feedback_end = time.perf_counter() + float(self.exp["feedback_s"])
            while time.perf_counter() < feedback_end:
                for _ in self._handle_control_events(allow_pause=False):
                    pass
                raw_mouse.drain()
                self.draw_inter_trial_cursor(cursor)
                pygame.display.flip()
                self.clock.tick(120)
        return TrialRunResult(valid=valid, outcome=outcome, cursor=next_cursor.copy(), trial_stage=trial_stage)

    def run(self):
        self.initialize_display()
        metadata = self.session_metadata()
        logger = AsyncExperimentLogger(
            self.output_dir, metadata, int(self.config["logging"]["writer_queue_size"])
        )
        raw_mouse = (
            RawInputMouse()
            if self.input_backend == BACKEND_WINDOWS_RAW_INPUT
            else PygameRelativeMouse()
        )
        aborted = False
        try:
            raw_mouse.start()
            self.instruction_screen(self.main_instruction_lines())
            practice_n = int(self.exp["practice_trials_per_cpi"])
            measured_n = int(self.exp["measured_trials_per_cpi"])
            practice_templates = self.templates[:practice_n]
            measured_templates = self.templates[practice_n:practice_n + measured_n]

            for block_index, cpi in enumerate(self.cpi_order, start=1):
                self.instruction_screen([
                    f"Block {block_index}/{len(self.cpi_order)} - {cpi} CPI",
                    f"Set the mouse hardware to {cpi} CPI.",
                    f"Mousepad condition: {self.mousepad_condition}.",
                    "Do not change the Windows pointer sensitivity.",
                    "Press SPACE when you are ready.",
                ])
                logger.submit_event({
                    "event": "block_start", "session_id": self.session_id,
                    "block_index": block_index, "cpi": cpi,
                    "mousepad_condition": self.mousepad_condition,
                    "timestamp_ns": time.perf_counter_ns(),
                    "input_backend": self.input_backend,
                })
                for phase, phase_templates in (
                    ("practice", practice_templates), ("measured", measured_templates)
                ):
                    if phase == "measured":
                        self.instruction_screen([
                            f"The {cpi} CPI practice phase is complete.",
                            f"The {measured_n} recorded trials will now begin.",
                            "Press SPACE to continue.",
                        ])
                    for phase_trial_index, template in enumerate(phase_templates, start=1):
                        phase_trial_total = len(phase_templates)
                        attempt = 1
                        while True:
                            result = None
                            try:
                                if bool(self.exp.get("start_target_enabled", False)):
                                    start_result = self.run_trial(
                                        template, cpi, block_index, phase, attempt,
                                        phase_trial_index, phase_trial_total,
                                        raw_mouse, logger, trial_stage="start",
                                    )
                                    result = start_result
                                    if start_result.valid and start_result.outcome == "success":
                                        result = self.run_trial(
                                            template, cpi, block_index, phase, attempt,
                                            phase_trial_index, phase_trial_total,
                                            raw_mouse, logger, trial_stage="task",
                                        )
                                    valid = bool(
                                        result.valid
                                        and (
                                            result.trial_stage == "task"
                                            or result.outcome == "success"
                                        )
                                    )
                                else:
                                    result = self.run_trial(
                                        template, cpi, block_index, phase, attempt,
                                        phase_trial_index, phase_trial_total,
                                        raw_mouse, logger, trial_stage="task",
                                    )
                                    valid = result.valid
                            except ExperimentPaused as pause:
                                self.pause_menu(logger, raw_mouse)
                                if pause.trial_logged:
                                    attempt += 1
                                continue
                            if valid:
                                break
                            if result and result.trial_stage == "start" and result.valid:
                                self.instruction_screen([
                                    "This trial was missed.",
                                    "Please click inside the target.",
                                    "Press SPACE to repeat the trial.",
                                ])
                            else:
                                self.instruction_screen([
                                    "This trial was invalidated because the window lost focus.",
                                    "The trial will be repeated. Press SPACE to continue.",
                                ])
                            attempt += 1
                logger.submit_event({
                    "event": "block_end", "session_id": self.session_id,
                    "block_index": block_index, "cpi": cpi,
                    "mousepad_condition": self.mousepad_condition,
                    "timestamp_ns": time.perf_counter_ns(),
                    "input_backend": self.input_backend,
                })
            self.instruction_screen([
                "The experiment is complete. Thank you!",
                "Press SPACE to close.",
            ])
        except ExperimentAborted:
            aborted = True
            logger.submit_event({
                "event": "session_aborted", "session_id": self.session_id,
                "timestamp_ns": time.perf_counter_ns(),
                "input_backend": self.input_backend,
            })
        finally:
            raw_mouse.stop()
            logger.submit_event({
                "event": "session_end", "session_id": self.session_id,
                "timestamp_ns": time.perf_counter_ns(), "aborted": aborted,
                "logger_backpressure_count": logger.queue_backpressure_count,
                "input_backend": self.input_backend,
            })
            logger.close()
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
            pygame.quit()
        return 130 if aborted else 0


def validate(config: dict[str, Any], participant: str) -> None:
    exp = config["experiment"]
    virtual_size = tuple(float(v) for v in config["display"]["virtual_size_m"])
    n = int(exp["practice_trials_per_cpi"]) + int(exp["measured_trials_per_cpi"])
    templates = make_trial_templates(
        n,
        virtual_size,
        tuple(exp["target_id_range"]),
        int(exp["seed"]),
        cursor_position_m=None,
        target_radius_range_m=tuple(
            float(v)
            for v in exp.get("target_radius_range_m", (0.00125, 0.007))
        ),
    )
    order = counterbalanced_cpi_order(participant, exp["cpi_values"], int(exp["seed"]))
    assert len(templates) == n
    assert all(exp["target_id_range"][0] <= t.fitts_id <= exp["target_id_range"][1] for t in templates)
    print(json.dumps({
        "status": "ok", "templates": len(templates), "cpi_order": order,
        "radius_range_m": [min(t.target_radius_m for t in templates),
                           max(t.target_radius_m for t in templates)],
    }, indent=2))


def validate_logger() -> None:
    """Small end-to-end spawn/write check, used only by the validation command."""
    with tempfile.TemporaryDirectory(prefix="pnc_logger_test_") as temp_dir:
        output = Path(temp_dir) / "session"
        logger = AsyncExperimentLogger(output, {"self_test": True}, queue_size=2)
        logger.submit_trial(
            {"session_id": "test", "trial_id": 1, "outcome": "success"}, [], [], []
        )
        logger.close()
        rows = (output / "trials.csv").read_text(encoding="utf-8").splitlines()
        if len(rows) != 2:
            raise RuntimeError("Logger validation failed")
    print("async_logger: ok")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", default="TEST")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default="data/human_experiment")
    parser.add_argument("--age", type=int)
    parser.add_argument("--sex", default="not_reported")
    parser.add_argument("--dominant-hand", choices=["right", "left", "ambidextrous"], default="right")
    parser.add_argument("--mouse-experience", default="not_reported")
    parser.add_argument("--gaming-hours", type=float)
    parser.add_argument("--mouse-model", default="not_reported")
    parser.add_argument(
        "--mousepad-condition",
        choices=("firm", "soft", "glass", "unspecified"),
        default="unspecified",
    )
    parser.add_argument("--os-pointer-speed", default="not_reported")
    parser.add_argument("--pointer-acceleration", choices=["on", "off", "unknown"], default="unknown")
    parser.add_argument("--notes", default="")
    parser.add_argument("--windowed", action="store_true", help="Debug only; never use for collection")
    parser.add_argument(
        "--input-backend",
        choices=("auto", BACKEND_WINDOWS_RAW_INPUT, BACKEND_PYGAME_RELATIVE),
        default="auto",
        help=(
            "auto uses Windows Raw Input on Windows and pygame_relative on "
            "macOS/Linux. pygame_relative is for pilot/debug only."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-logger", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.validate_only:
        validate(config, args.participant)
        if args.validate_logger:
            validate_logger()
        return 0
    return PointAndClickExperiment(args, config).run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
