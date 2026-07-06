"""Browser-based visualizer for a trained PnC SAC policy."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from stable_baselines3 import SAC

from .visualize_policy import (
    BEST_POLICY_ACCURACY,
    PROJECT_ROOT,
    build_environment,
    load_slider_profile,
    record_episode,
    resolve_model_path,
)


HTML_PATH = Path(__file__).resolve().parent / "web" / "visualize_policy.html"
SPRITE_PATH = PROJECT_ROOT / "src" / "materials" / "mouse_hand_sprite.png"

PARAMETER_LABELS = {
    "theta_m": "Motor noise",
    "theta_c": "Timing noise",
    "rew_succ": "Success reward",
    "rew_fail": "Failure penalty",
    "time_penalty": "Time cost",
    "effort_cost": "Effort cost",
    "mouse_rotation_gain_x": "Rotation gain x",
    "mouse_rotation_gain_y": "Rotation gain y",
    "mouse_plan_correction": "Plan correction",
}


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class PolicyWebService:
    """Own the model and its mutable environment for the HTTP handlers."""

    def __init__(self, args: argparse.Namespace):
        self.model_path = resolve_model_path(args.model, args.ckpt)
        self.env = build_environment(
            self.model_path,
            env_class_name=args.env_class_type,
            agent_type=args.agent_type,
            pnc_type=args.pnc_type,
        )
        self.model = SAC.load(str(self.model_path), device=args.device)
        if self.model.observation_space.shape != self.env.observation_space.shape:
            raise ValueError(
                f"Model observation shape {self.model.observation_space.shape} does not "
                f"match environment shape {self.env.observation_space.shape}."
            )
        self.defaults = load_slider_profile(self.model_path, args.slider_profile)
        self.deterministic = not args.stochastic
        self.lock = threading.Lock()
        self.parameter_names = [
            name for name in PARAMETER_LABELS if name in list(self.env.user.param_modul.list)
        ]
        self.slider_specs = self._make_slider_specs(args.seed)

    @staticmethod
    def _step_for(minimum: float, maximum: float, name: str) -> float:
        if name == "seed":
            return 1.0
        if name == "target_direction":
            return 0.1
        span = maximum - minimum
        return max(span / 1000.0, 1e-8)

    def _make_slider_specs(self, seed: int) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for name in self.parameter_names:
            parameter_range = self.env.user.param_modul[name]
            minimum = float(parameter_range.min)
            maximum = float(parameter_range.max)
            initial = float(self.defaults.get(name, (minimum + maximum) / 2))
            specs.append(
                {
                    "name": name,
                    "label": PARAMETER_LABELS[name],
                    "min": minimum,
                    "max": maximum,
                    "step": self._step_for(minimum, maximum, name),
                    "value": float(np.clip(initial, minimum, maximum)),
                }
            )

        radius_min, radius_max = map(float, self.env.pnc_env.target_rad_range)
        task_specs = (
            (
                "target_radius",
                "Target radius",
                radius_min,
                radius_max,
                self.defaults.get("target_radius", (radius_min + radius_max) / 2),
            ),
            ("target_speed", "Target speed", 0.0, 0.24, self.defaults.get("target_speed", 0.0)),
            (
                "target_direction",
                "Target direction",
                0.0,
                360.0,
                self.defaults.get("target_direction", 0.0),
            ),
            ("seed", "Random seed", 0.0, 999.0, self.defaults.get("seed", seed)),
        )
        for name, label, minimum, maximum, initial in task_specs:
            specs.append(
                {
                    "name": name,
                    "label": label,
                    "min": minimum,
                    "max": maximum,
                    "step": self._step_for(minimum, maximum, name),
                    "value": float(np.clip(initial, minimum, maximum)),
                }
            )
        return specs

    def config(self) -> dict[str, Any]:
        return {
            "modelName": self.model_path.parent.parent.name,
            "modelPath": str(self.model_path),
            "sliders": self.slider_specs,
            "profile": self.defaults.get("_metadata", {}),
            "bestPolicyAccuracy": BEST_POLICY_ACCURACY,
        }

    def _bounded_values(self, supplied: dict[str, Any]) -> dict[str, float]:
        values: dict[str, float] = {}
        for spec in self.slider_specs:
            name = spec["name"]
            try:
                value = float(supplied.get(name, spec["value"]))
            except (TypeError, ValueError):
                value = float(spec["value"])
            values[name] = float(np.clip(value, spec["min"], spec["max"]))
        values["seed"] = float(round(values["seed"]))
        return values

    def _apply_values(self, values: dict[str, float]) -> None:
        normalized_parameters = {}
        for name in self.parameter_names:
            parameter_range = self.env.user.param_modul[name]
            minimum = float(parameter_range.min)
            maximum = float(parameter_range.max)
            normalized_parameters[name] = float(
                np.clip((2 * values[name] - minimum - maximum) / (maximum - minimum), -1, 1)
            )
        self.env.fix_z(normalized_parameters)
        self.env.update_task_setting(
            {
                "target_rad": values["target_radius"],
                "target_spd": values["target_speed"],
                "target_dir": values["target_direction"],
            }
        )

    def rollout(self, supplied: dict[str, Any]) -> dict[str, Any]:
        values = self._bounded_values(supplied)
        with self.lock:
            self._apply_values(values)
            frames, total_reward = record_episode(
                self.model,
                self.env,
                deterministic=self.deterministic,
                seed=int(values["seed"]),
            )

            cursor = np.stack([frame.cursor_pos for frame in frames])
            target = np.stack([frame.target_pos for frame in frames])
            hand = np.stack([frame.hand_pos for frame in frames])
            cursor_reference = cursor[0] + (hand - hand[0]) * float(self.env.pnc_env.mouse_gain)
            distance = np.linalg.norm(cursor - target, axis=1)
            times = np.asarray([frame.time_s for frame in frames], dtype=float)
            times -= times[0]

        frame_data = []
        for index, frame in enumerate(frames):
            frame_data.append(
                {
                    "time": float(times[index]),
                    "cursor": cursor[index].tolist(),
                    "cursorReference": cursor_reference[index].tolist(),
                    "target": target[index].tolist(),
                    "targetRadius": float(frame.target_radius),
                    "hand": hand[index].tolist(),
                    "rotation": float(frame.mouse_orientation),
                    "distance": float(distance[index]),
                    "reward": float(frame.reward),
                    "done": bool(frame.done),
                    "success": bool(frame.success),
                    "truncated": bool(frame.truncated),
                    "click": None if frame.click_pos is None else frame.click_pos.tolist(),
                }
            )

        final = frames[-1]
        click_distance = float(distance[-1])
        mouse_path = float(np.linalg.norm(np.diff(hand, axis=0), axis=1).sum())
        return {
            "values": values,
            "frames": frame_data,
            "metrics": {
                "hit": bool(final.success),
                "truncated": bool(final.truncated),
                "episodeTime": float(times[-1]),
                "totalReward": float(total_reward),
                "clickDistance": click_distance,
                "clickRadiusRatio": click_distance / max(float(final.target_radius), 1e-12),
                "targetRadius": float(final.target_radius),
                "targetSpeed": values["target_speed"],
                "targetDirection": values["target_direction"],
                "mousePath": mouse_path,
                "finalRotationDegrees": float(np.degrees(final.mouse_orientation)),
            },
        }


def make_handler(service: PolicyWebService):
    class PolicyRequestHandler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: Any) -> None:
            self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlparse(self.path).path
            if path == "/":
                self._send(HTTPStatus.OK, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/config":
                self._json(HTTPStatus.OK, service.config())
            elif path == "/sprite.png":
                self._send(HTTPStatus.OK, SPRITE_PATH.read_bytes(), "image/png")
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if urlparse(self.path).path != "/api/rollout":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 1_000_000)
                body = json.loads(self.rfile.read(length) or b"{}")
                values = body.get("parameters", {})
                if not isinstance(values, dict):
                    raise ValueError("parameters must be an object")
                self._json(HTTPStatus.OK, service.rollout(values))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # Keep the local UI informative.
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return PolicyRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the PnC policy visualizer in a browser.")
    parser.add_argument("--model", default=None, help="Run name, run directory, or SAC .zip path")
    parser.add_argument("--ckpt", default="last", help="'last', 'best', or an integer checkpoint step")
    parser.add_argument("--slider-profile", choices=("last", "best", "midpoint"), default="last")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--env_class_type", default="PnCEnvDefault")
    parser.add_argument("--agent_type", default="classic")
    parser.add_argument("--pnc_type", default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the page automatically")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = PolicyWebService(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"PnC policy visualizer: {url}")
    print(f"Model: {service.model_path}")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
