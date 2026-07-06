"""Visualize a trained PnC SAC policy on the monitor and the desk."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from box import Box
from stable_baselines3 import SAC

from ..agent import pnc_agent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "data" / "pnc_model"
BEST_POLICY_ACCURACY = 0.154  # 500 deterministic episodes over the training distribution


@dataclass
class PolicyFrame:
    time_s: float
    cursor_pos: np.ndarray
    target_pos: np.ndarray
    target_radius: float
    hand_pos: np.ndarray
    mouse_orientation: float
    window_half_size: np.ndarray
    reward: float = 0.0
    action: Optional[np.ndarray] = None
    action_values: Optional[dict] = None
    done: bool = False
    success: bool = False
    truncated: bool = False
    click_pos: Optional[np.ndarray] = None


def find_latest_model(model_root: Path = MODEL_ROOT) -> Path:
    """Return the newest last model, falling back to checkpoints and then best models."""
    if not model_root.exists():
        candidates = []
    else:
        last_models = list(model_root.rglob("last_model.zip"))
        checkpoints = [
            path
            for path in model_root.rglob("rl_model_*_steps.zip")
            if _checkpoint_step(path) >= 0
        ]
        best_models = list(model_root.rglob("best_model.zip"))
        candidates = last_models or checkpoints or best_models or list(model_root.rglob("*.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No SAC model found below {model_root}. Pass one explicitly with --model."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _checkpoint_step(path: Path) -> int:
    name = path.stem
    prefix = "rl_model_"
    suffix = "_steps"
    if name.startswith(prefix) and name.endswith(suffix):
        try:
            return int(name[len(prefix):-len(suffix)])
        except ValueError:
            return -1
    return -1


def resolve_model_path(model: Optional[str], checkpoint: str = "last") -> Path:
    """Resolve a run name/directory to its last model, best model, or a numbered checkpoint."""
    if model is None:
        return find_latest_model().resolve()

    supplied_path = Path(model).expanduser()
    if supplied_path.is_file():
        return supplied_path.resolve()

    run_directory = supplied_path if supplied_path.is_dir() else MODEL_ROOT / model
    if not run_directory.is_dir():
        available_runs = sorted(path.name for path in MODEL_ROOT.iterdir() if path.is_dir())
        raise FileNotFoundError(
            f"Unknown model run: {model}. Available runs: {', '.join(available_runs)}"
        )

    checkpoint = checkpoint.lower()
    if checkpoint == "best":
        model_path = run_directory / "best" / "best_model.zip"
    elif checkpoint == "last":
        saved_final_model = run_directory / "last" / "last_model.zip"
        checkpoint_candidates = [
            path
            for path in (run_directory / "checkpoint").glob("rl_model_*_steps.zip")
            if _checkpoint_step(path) >= 0
        ]
        if saved_final_model.is_file():
            model_path = saved_final_model
        elif checkpoint_candidates:
            model_path = max(checkpoint_candidates, key=_checkpoint_step)
        else:
            model_path = run_directory / "best" / "best_model.zip"
    else:
        try:
            checkpoint_step = int(checkpoint)
        except ValueError as error:
            raise ValueError("--ckpt must be 'last', 'best', or an integer step count") from error
        model_path = run_directory / "checkpoint" / f"rl_model_{checkpoint_step}_steps.zip"

    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    return model_path.resolve()


def find_run_directory(model_path: Path) -> Optional[Path]:
    for parent in model_path.resolve().parents:
        if (parent / "env_info.yaml").is_file():
            return parent
        if parent == PROJECT_ROOT:
            break
    return None


def load_slider_profile(model_path: Path, profile: str = "last") -> dict:
    """Load model-specific visualizer defaults saved beside a training run."""
    if profile == "midpoint":
        return {"_metadata": {"profile": "neutral", "reason": "requested"}}
    run_directory = find_run_directory(model_path)
    if run_directory is None:
        return {"_metadata": {"profile": "neutral", "reason": "run not found"}}
    if profile == "last":
        profile_paths = (
            run_directory / "visualizer_last_params.yaml",
            run_directory / "last" / "visualizer_last_params.yaml",
        )
        expected_filename = "visualizer_last_params.yaml"
    elif profile == "best":
        profile_paths = (
            run_directory / "best" / "visualizer_best_params.yaml",
            run_directory / "visualizer_best_params.yaml",  # Legacy runs
        )
        expected_filename = "visualizer_best_params.yaml"
    else:
        return {
            "_metadata": {
                "profile": "neutral",
                "reason": f"unknown slider profile: {profile}",
            }
        }
    profile_path = next((path for path in profile_paths if path.is_file()), None)
    if profile_path is None:
        return {
            "_metadata": {
                "profile": "neutral",
                "reason": f"{expected_filename} not found",
            }
        }
    try:
        with profile_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        return {
            "_metadata": {
                "profile": "neutral",
                "reason": "saved profile could not be read",
            }
        }
    if not isinstance(data, dict):
        return {
            "_metadata": {
                "profile": "neutral",
                "reason": "saved profile is invalid",
            }
        }
    parameters = dict(data.get("parameters") or {})
    return {
        **parameters,
        **(data.get("task") or {}),
        "_metadata": {
            key: data.get(key)
            for key in (
                "profile",
                "checkpoint",
                "criterion",
                "evaluation_timesteps",
                "evaluation_episodes",
                "evaluation_mean_reward",
                "measured_success_rate",
                "selected_episode_reward",
                "selected_episode_success",
            )
        },
    }


def build_environment(
    model_path: Path,
    env_class_name: str = "PnCEnvDefault",
    agent_type: str = "classic",
    pnc_type: str = "default",
):
    """Recreate the environment saved beside the checkpoint when possible."""
    run_directory = find_run_directory(model_path)
    env_info_path = run_directory / "env_info.yaml" if run_directory else None

    if env_info_path and env_info_path.is_file():
        with env_info_path.open("r", encoding="utf-8") as stream:
            env_info = yaml.safe_load(stream)
        env_class_name = env_info.get("env_class", env_class_name)
        env_class = getattr(pnc_agent, env_class_name)
        return env_class(
            agent_cfg=Box(env_info["user_config"]),
            pnc_env_cfg=Box(env_info["task_config"]),
            interval_cfg=Box(env_info["interval"]),
        )

    env_class = getattr(pnc_agent, env_class_name)
    return env_class(agent_name=agent_type, pnc_env_name=pnc_type)


def _snapshot(env, **overrides) -> PolicyFrame:
    task = env.pnc_env
    values = dict(
        time_s=env.timer / 1000,
        cursor_pos=np.asarray(task.cursor_pos, dtype=float).copy(),
        target_pos=np.asarray(task.target_pos, dtype=float).copy(),
        target_radius=float(task.target_rad),
        hand_pos=np.asarray(task.hand_pos, dtype=float).copy(),
        mouse_orientation=float(task.mouse_ori),
        window_half_size=np.asarray(task.window_qt, dtype=float).copy(),
    )
    values.update(overrides)
    return PolicyFrame(**values)


def record_episode(model: SAC, env, deterministic: bool = True, seed: int = 0):
    """Run one episode and retain the physical state after each policy action."""
    np.random.seed(seed)
    env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    frames = [_snapshot(env)]
    total_reward = 0.0

    while True:
        action, _ = model.predict(observation, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32)
        action_values = env._unpack_action(action)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)

        click_pos = None
        step_state = info.get("step_state")
        if done and step_state is not None and "click_endpt" in step_state:
            click_pos = np.asarray(step_state.click_endpt, dtype=float).copy()

        frames.append(
            _snapshot(
                env,
                reward=float(reward),
                action=action.copy(),
                action_values=dict(action_values),
                done=done,
                success=bool(info.get("is_success", False)),
                truncated=bool(truncated),
                click_pos=click_pos,
            )
        )
        if done:
            return frames, total_reward


def create_animation(frames, model_name: str, fps: float = 10.0, repeat: bool = True):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle, Rectangle

    figure, (monitor_axis, desk_axis) = plt.subplots(1, 2, figsize=(12, 5.8))
    figure.patch.set_facecolor("#111827")
    figure.canvas.manager.set_window_title("PnC optimized policy")

    for axis in (monitor_axis, desk_axis):
        axis.set_facecolor("#0b1220")
        axis.tick_params(colors="#94a3b8")
        for spine in axis.spines.values():
            spine.set_color("#334155")
        axis.grid(color="#1e293b", linewidth=0.6, alpha=0.55)
        axis.set_aspect("equal", adjustable="box")

    half_width, half_height = frames[0].window_half_size
    monitor_axis.set_xlim(-half_width, half_width)
    monitor_axis.set_ylim(-half_height, half_height)
    monitor_axis.set_title("MONITOR", color="#e2e8f0", weight="bold")
    monitor_axis.set_xlabel("x (m)", color="#94a3b8")
    monitor_axis.set_ylabel("y (m)", color="#94a3b8")
    monitor_axis.add_patch(
        Rectangle(
            (-half_width, -half_height),
            2 * half_width,
            2 * half_height,
            fill=False,
            edgecolor="#64748b",
            linewidth=2,
        )
    )

    hand_positions = np.stack([frame.hand_pos for frame in frames])
    desk_min = hand_positions.min(axis=0)
    desk_max = hand_positions.max(axis=0)
    desk_center = (desk_min + desk_max) / 2
    desk_span = np.maximum(desk_max - desk_min, np.array([0.18, 0.18]))
    desk_margin = np.maximum(0.04, desk_span * 0.25)
    desk_axis.set_xlim(
        desk_center[0] - desk_span[0] / 2 - desk_margin[0],
        desk_center[0] + desk_span[0] / 2 + desk_margin[0],
    )
    desk_axis.set_ylim(
        desk_center[1] - desk_span[1] / 2 - desk_margin[1],
        desk_center[1] + desk_span[1] / 2 + desk_margin[1],
    )
    desk_axis.set_title("DESK / HAND", color="#e2e8f0", weight="bold")
    desk_axis.set_xlabel("x (m)", color="#94a3b8")
    desk_axis.set_ylabel("y (m)", color="#94a3b8")

    target = Circle(
        frames[0].target_pos,
        frames[0].target_radius,
        facecolor="#f59e0b",
        edgecolor="#fde68a",
        linewidth=1.6,
        alpha=0.9,
        zorder=4,
    )
    monitor_axis.add_patch(target)
    cursor, = monitor_axis.plot(
        [], [], marker="+", markersize=15, markeredgewidth=2.2,
        color="#22d3ee", linestyle="None", zorder=6, label="Cursor"
    )
    cursor_trail, = monitor_axis.plot([], [], color="#0891b2", linewidth=1.4, alpha=0.65)
    target_trail, = monitor_axis.plot([], [], color="#d97706", linewidth=1.0, alpha=0.45)
    click_marker, = monitor_axis.plot(
        [], [], marker="x", markersize=11, markeredgewidth=2,
        color="#f8fafc", linestyle="None", zorder=7
    )

    hand, = desk_axis.plot(
        [], [], marker="o", markersize=10, color="#f472b6",
        markeredgecolor="#fce7f3", linestyle="None", zorder=6, label="Hand"
    )
    hand_trail, = desk_axis.plot([], [], color="#db2777", linewidth=1.8, alpha=0.7)
    orientation_line, = desk_axis.plot([], [], color="#fbcfe8", linewidth=2.4, zorder=7)

    status_text = figure.text(
        0.5, 0.035, "", ha="center", va="center", color="#cbd5e1", fontsize=10
    )
    figure.suptitle(
        f"Optimized SAC policy — {model_name}",
        color="#f8fafc", fontsize=14, weight="bold", y=0.97,
    )
    figure.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.88, wspace=0.25)

    cursor_positions = np.stack([frame.cursor_pos for frame in frames])
    target_positions = np.stack([frame.target_pos for frame in frames])

    def update(frame_index):
        frame = frames[frame_index]
        target.center = frame.target_pos
        target.radius = frame.target_radius
        target.set_facecolor("#22c55e" if frame.done and frame.success else "#f59e0b")
        cursor.set_data([frame.cursor_pos[0]], [frame.cursor_pos[1]])
        cursor_trail.set_data(
            cursor_positions[:frame_index + 1, 0], cursor_positions[:frame_index + 1, 1]
        )
        target_trail.set_data(
            target_positions[:frame_index + 1, 0], target_positions[:frame_index + 1, 1]
        )

        if frame.click_pos is None:
            click_marker.set_data([], [])
        else:
            click_marker.set_data([frame.click_pos[0]], [frame.click_pos[1]])

        hand.set_data([frame.hand_pos[0]], [frame.hand_pos[1]])
        hand_trail.set_data(
            hand_positions[:frame_index + 1, 0], hand_positions[:frame_index + 1, 1]
        )
        orientation_length = 0.035
        orientation_end = frame.hand_pos + orientation_length * np.array(
            [np.cos(frame.mouse_orientation), np.sin(frame.mouse_orientation)]
        )
        orientation_line.set_data(
            [frame.hand_pos[0], orientation_end[0]],
            [frame.hand_pos[1], orientation_end[1]],
        )

        if frame.action_values is None:
            action_description = "initial state"
        else:
            action_description = (
                f"plan={frame.action_values['th']:.0f} ms  "
                f"click={int(frame.action_values['kc'])}  "
                f"timing={frame.action_values['tc']:.2f}"
            )
        outcome = ""
        if frame.done:
            outcome = "  •  HIT" if frame.success else "  •  MISS"
        status_text.set_text(
            f"t={frame.time_s:.2f}s  •  step {frame_index}/{len(frames)-1}  •  "
            f"reward={frame.reward:+.3f}  •  {action_description}{outcome}"
        )
        return (
            target, cursor, cursor_trail, target_trail, click_marker,
            hand, hand_trail, orientation_line, status_text,
        )

    return figure, FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
        repeat=repeat,
    )


def create_parameter_visualizer(
    model: SAC,
    env,
    model_name: str,
    seed: int = 0,
    fps: float = 10.0,
    deterministic: bool = True,
    interactive: bool = True,
    slider_defaults: Optional[dict] = None,
):
    """Build the interactive dashboard used to inspect a trained policy."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.transforms import Affine2D
    from matplotlib.widgets import Button, Slider

    theme = {
        "figure": "#111827",
        "panel": "#172033",
        "panel_alt": "#0f172a",
        "slider_bg": "#1f2937",
        "button": "#334155",
        "button_hover": "#475569",
        "grid": "#334155",
        "spine": "#475569",
        "text": "#e5e7eb",
        "muted": "#94a3b8",
        "cursor": "#38bdf8",
        "cursor_reference": "#94a3b8",
        "target": "#fb7185",
        "click": "#f97316",
        "distance": "#f8fafc",
        "rotation": "#22d3ee",
        "desk": "#7c5f46",
        "desk_edge": "#3f2a1d",
        "mousepad": "#1f2937",
        "accent": "#60a5fa",
    }

    def style_axis(axis, *, grid: bool = True):
        axis.set_facecolor(theme["panel"])
        axis.title.set_color(theme["text"])
        axis.xaxis.label.set_color(theme["muted"])
        axis.yaxis.label.set_color(theme["muted"])
        axis.tick_params(colors=theme["muted"])
        for spine in axis.spines.values():
            spine.set_color(theme["spine"])
        if grid:
            axis.grid(True, color=theme["grid"], linewidth=0.65, alpha=0.38)

    figure = plt.figure(figsize=(14.4, 8.6), facecolor=theme["figure"])
    figure.patch.set_facecolor(theme["figure"])
    if hasattr(figure.canvas.manager, "set_window_title"):
        figure.canvas.manager.set_window_title("PnC policy parameter visualizer")
    scene_axis = figure.add_axes([0.055, 0.43, 0.34, 0.50])
    desk_axis = figure.add_axes([0.43, 0.43, 0.18, 0.50])
    distance_axis = figure.add_axes([0.665, 0.62, 0.29, 0.29])
    rotation_axis = figure.add_axes([0.665, 0.41, 0.29, 0.13], sharex=distance_axis)
    metrics_axis = figure.add_axes([0.665, 0.12, 0.29, 0.20])
    metrics_axis.set_facecolor(theme["panel_alt"])
    metrics_axis.axis("off")
    figure.suptitle(f"SAC policy — {model_name}", fontsize=14, weight="bold")

    if figure._suptitle is not None:
        figure._suptitle.set_color(theme["text"])

    parameter_names = [
        "theta_m",
        "theta_c",
        "rew_succ",
        "rew_fail",
        "time_penalty",
        "effort_cost",
        "mouse_rotation_gain_x",
        "mouse_rotation_gain_y",
        "mouse_plan_correction",
    ]
    parameter_labels = {
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
    slider_defaults = slider_defaults or {}
    slider_specs = []
    modul_names = list(env.user.param_modul.list)
    for name in parameter_names:
        if name not in modul_names:
            continue
        parameter_range = env.user.param_modul[name]
        minimum = float(parameter_range.min)
        maximum = float(parameter_range.max)
        initial = slider_defaults.get(name, (minimum + maximum) / 2)
        initial = float(np.clip(initial, minimum, maximum))
        value_format = "%.3f" if maximum - minimum <= 2 else "%.1f"
        slider_specs.append(
            (name, parameter_labels[name], minimum, maximum, initial, value_format)
        )

    target_radius_min, target_radius_max = env.pnc_env.target_rad_range
    slider_specs.extend(
        [
            (
                "target_radius",
                "Target radius",
                float(target_radius_min),
                float(target_radius_max),
                float(
                    np.clip(
                        slider_defaults.get(
                            "target_radius", (target_radius_min + target_radius_max) / 2
                        ),
                        target_radius_min,
                        target_radius_max,
                    )
                ),
                "%.4f",
            ),
            (
                "target_speed", "Target speed", 0.0, 0.24,
                float(np.clip(slider_defaults.get("target_speed", 0.0), 0.0, 0.24)), "%.3f"
            ),
            (
                "target_direction", "Target direction", 0.0, 360.0,
                float(np.clip(slider_defaults.get("target_direction", 0.0), 0.0, 360.0)), "%.0f"
            ),
            (
                "seed", "Random seed", 0.0, 999.0,
                float(np.clip(slider_defaults.get("seed", seed), 0.0, 999.0)), "%.0f"
            ),
        ]
    )

    sliders = {}
    rows_per_column = 7
    start_y = 0.285
    row_spacing = 0.032
    for index, (name, label, minimum, maximum, initial, value_format) in enumerate(slider_specs):
        column = index // rows_per_column
        row = index % rows_per_column
        slider_axis = figure.add_axes(
            [0.13 + column * 0.30, start_y - row * row_spacing, 0.17, 0.018]
        )
        slider_axis.set_facecolor(theme["slider_bg"])
        sliders[name] = Slider(
            slider_axis,
            label,
            minimum,
            maximum,
            valinit=initial,
            valfmt=value_format,
            color=theme["accent"],
            track_color="#334155",
            handle_style={
                "facecolor": theme["text"],
                "edgecolor": theme["accent"],
                "size": 7,
            },
        )
        sliders[name].label.set_fontsize(7.5)
        sliders[name].label.set_color(theme["muted"])
        sliders[name].valtext.set_color(theme["text"])
        for spine in slider_axis.spines.values():
            spine.set_color(theme["spine"])

    run_axis = figure.add_axes([0.665, 0.035, 0.08, 0.045])
    play_axis = figure.add_axes([0.76, 0.035, 0.10, 0.045])
    reset_axis = figure.add_axes([0.875, 0.035, 0.08, 0.045])
    run_button = Button(run_axis, "Run", color=theme["button"], hovercolor=theme["button_hover"])
    play_button = Button(play_axis, "Pause", color=theme["button"], hovercolor=theme["button_hover"])
    reset_button = Button(reset_axis, "Reset", color=theme["button"], hovercolor=theme["button_hover"])
    for button_axis, button in (
        (run_axis, run_button),
        (play_axis, play_button),
        (reset_axis, reset_button),
    ):
        button_axis.set_facecolor(theme["button"])
        for spine in button_axis.spines.values():
            spine.set_color(theme["spine"])
        button.label.set_color(theme["text"])
        button.label.set_fontweight("bold")

    scene_axis.set_title("Rollout video")
    scene_axis.set_xlabel("x (m)")
    scene_axis.set_ylabel("y (m)")
    scene_axis.set_aspect("equal", adjustable="box")
    style_axis(scene_axis)
    cursor_reference_line, = scene_axis.plot(
        [], [], color=theme["cursor_reference"], lw=1.0, alpha=0.65, ls="--", label="cursor without rotation"
    )
    cursor_line, = scene_axis.plot([], [], color=theme["cursor"], lw=1.8, label="screen cursor")
    target_line, = scene_axis.plot([], [], color=theme["target"], lw=1.4, alpha=0.8, label="target")
    cursor_point, = scene_axis.plot([], [], "o", color="#e0f2fe", ms=6)
    target_point, = scene_axis.plot([], [], "o", color=theme["target"], ms=4)
    click_point, = scene_axis.plot(
        [], [], marker="x", color=theme["click"], ms=9, mew=2, linestyle="None"
    )
    target_circle = Circle(
        (0, 0), 0.01, edgecolor=theme["target"], facecolor=theme["target"], alpha=0.22, lw=1.5
    )
    scene_axis.add_patch(target_circle)
    scene_legend = scene_axis.legend(loc="upper left", fontsize=8, frameon=False)
    for text in scene_legend.get_texts():
        text.set_color(theme["muted"])

    desk_axis.set_title("Table / physical mouse")
    desk_axis.set_xlabel("x (m)")
    desk_axis.set_ylabel("y (m)")
    desk_axis.set_aspect("equal", adjustable="box")
    style_axis(desk_axis)
    table_width = 0.60
    table_height = 0.40
    mousepad_width = 0.45
    mousepad_height = 0.30
    desk_axis.set_xlim(-table_width / 2, table_width / 2)
    desk_axis.set_ylim(-table_height / 2, table_height / 2)
    table_surface = Rectangle(
        (-table_width / 2, -table_height / 2),
        table_width,
        table_height,
        facecolor=theme["desk"],
        edgecolor=theme["desk_edge"],
        lw=2.0,
        zorder=-3,
    )
    mousepad_surface = Rectangle(
        (-mousepad_width / 2, -mousepad_height / 2),
        mousepad_width,
        mousepad_height,
        facecolor=theme["mousepad"],
        edgecolor="#0f172a",
        lw=1.5,
        zorder=-2,
    )
    desk_axis.add_patch(table_surface)
    desk_axis.add_patch(mousepad_surface)
    hand_trail_line, = desk_axis.plot(
        [], [], color="#a78bfa", lw=1.8, label="physical mouse path", zorder=3
    )
    sprite_path = PROJECT_ROOT / "src" / "materials" / "mouse_hand_sprite.png"
    sprite = plt.imread(sprite_path)
    if sprite.shape[-1] == 4:
        visible = sprite[..., 3] > 0.01
        rows, columns = np.nonzero(visible)
        sprite = sprite[rows.min():rows.max() + 1, columns.min():columns.max() + 1]

    # Realistic hand length, with the anchor at the physical mouse center
    # (roughly 20% down from the top of this top-down illustration).
    sprite_height = 0.24
    sprite_width = sprite_height * sprite.shape[1] / sprite.shape[0]
    mouse_anchor_from_top = 0.30
    sprite_extent = (
        -sprite_width / 2,
        sprite_width / 2,
        -sprite_height * (1 - mouse_anchor_from_top),
        sprite_height * mouse_anchor_from_top,
    )
    mouse_sprite = desk_axis.imshow(
        sprite,
        extent=sprite_extent,
        origin="upper",
        interpolation="bilinear",
        zorder=5,
    )
    mouse_center, = desk_axis.plot([], [], "o", color="#8b5cf6", ms=4, zorder=7)
    rotation_text = desk_axis.text(
        0.98, 0.96, "", ha="right", va="top", family="monospace", fontsize=9,
        transform=desk_axis.transAxes,
        color=theme["text"],
        bbox=dict(facecolor="#0f172a", edgecolor=theme["spine"], alpha=0.9, pad=4),
    )

    distance_axis.set_title("Distance to target")
    distance_axis.set_ylabel("distance (m)")
    style_axis(distance_axis)
    distance_axis.tick_params(labelbottom=False)
    distance_line, = distance_axis.plot([], [], color=theme["distance"], lw=1.6, label="distance")
    radius_line, = distance_axis.plot([], [], color=theme["target"], lw=1.2, ls="--", label="radius")
    distance_cursor = distance_axis.axvline(0, color=theme["accent"], lw=1)
    distance_legend = distance_axis.legend(loc="upper right", fontsize=8, frameon=False)
    for text in distance_legend.get_texts():
        text.set_color(theme["muted"])

    rotation_axis.set_title("Mouse rotation")
    rotation_axis.set_xlabel("time (s)")
    rotation_axis.set_ylabel("deg")
    style_axis(rotation_axis)
    rotation_line, = rotation_axis.plot([], [], color=theme["rotation"], lw=1.4)
    rotation_cursor = rotation_axis.axvline(0, color=theme["accent"], lw=1)
    metrics_text = metrics_axis.text(
        0, 1, "", va="top", ha="left", family="monospace", fontsize=9.5,
        color=theme["text"],
        transform=metrics_axis.transAxes,
    )

    state = {
        "frames": [],
        "frame": 0,
        "playing": True,
        "reward": 0.0,
        "data": None,
        "animation": None,
        "suppress_slider_events": False,
    }

    def slider_value(name):
        return float(sliders[name].val)

    def apply_parameters():
        normalized_parameters = {}
        for name in parameter_names:
            if name not in sliders:
                continue
            parameter_range = env.user.param_modul[name]
            value = slider_value(name)
            normalized_parameters[name] = float(
                np.clip(
                    (2 * value - (parameter_range.min + parameter_range.max))
                    / (parameter_range.max - parameter_range.min),
                    -1,
                    1,
                )
            )
        env.fix_z(normalized_parameters)
        env.update_task_setting(
            {
                "target_rad": slider_value("target_radius"),
                "target_spd": slider_value("target_speed"),
                "target_dir": slider_value("target_direction"),
            }
        )

    def collect_episode():
        apply_parameters()
        frames, total_reward = record_episode(
            model,
            env,
            deterministic=deterministic,
            seed=int(round(slider_value("seed"))),
        )
        cursor_positions = np.stack([frame.cursor_pos for frame in frames])
        target_positions = np.stack([frame.target_pos for frame in frames])
        hand_positions = np.stack([frame.hand_pos for frame in frames])
        times = np.array([frame.time_s for frame in frames])
        times -= times[0]
        distances = np.linalg.norm(cursor_positions - target_positions, axis=1)
        cursor_reference = (
            cursor_positions[0]
            + (hand_positions - hand_positions[0]) * float(env.pnc_env.mouse_gain)
        )
        state.update(
            frames=frames,
            frame=0,
            playing=True,
            reward=total_reward,
            data=dict(
                cursor=cursor_positions,
                target=target_positions,
                hand=hand_positions,
                time=times,
                distance=distances,
                cursor_reference=cursor_reference,
                rotation=np.array([frame.mouse_orientation for frame in frames]),
            ),
        )
        configure_episode_plots()
        update_frame(0)

    def configure_episode_plots():
        data = state["data"]
        frames = state["frames"]
        monitor_points = np.vstack((data["cursor"], data["cursor_reference"], data["target"]))
        low = monitor_points.min(axis=0) - frames[0].target_radius - 0.025
        high = monitor_points.max(axis=0) + frames[0].target_radius + 0.025
        center = (low + high) / 2
        half_span = max(float(np.max(high - low)) / 2, 0.04)
        scene_axis.set_xlim(center[0] - half_span, center[0] + half_span)
        scene_axis.set_ylim(center[1] - half_span, center[1] + half_span)

        # Keep a stable physical table frame. Expand it only if a trajectory
        # genuinely leaves the nominal 60 x 40 cm surface.
        hand_extent = np.max(np.abs(data["hand"]), axis=0) + np.array(
            [sprite_width / 2, sprite_height * (1 - mouse_anchor_from_top)]
        )
        x_half = max(table_width / 2, float(hand_extent[0]) * 1.10)
        y_half = max(table_height / 2, float(hand_extent[1]) * 1.10)
        desk_axis.set_xlim(-x_half, x_half)
        desk_axis.set_ylim(-y_half, y_half)

        distance_line.set_data(data["time"], data["distance"])
        radius_line.set_data(data["time"], np.full_like(data["time"], frames[0].target_radius))
        time_max = max(float(data["time"][-1]), 0.01)
        distance_axis.set_xlim(0, time_max)
        distance_axis.set_ylim(0, max(float(data["distance"].max()), frames[0].target_radius) * 1.15 + 1e-4)
        rotation_degrees = np.degrees(data["rotation"])
        rotation_line.set_data(data["time"], rotation_degrees)
        rotation_axis.set_xlim(0, time_max)
        rotation_min = float(rotation_degrees.min())
        rotation_max = float(rotation_degrees.max())
        if rotation_max - rotation_min < 1e-6:
            rotation_min -= 5
            rotation_max += 5
        rotation_axis.set_ylim(rotation_min - 3, rotation_max + 3)

        final_frame = frames[-1]
        click_distance = float(data["distance"][-1])
        mouse_path = float(np.linalg.norm(np.diff(data["hand"], axis=0), axis=1).sum())
        profile_metadata = slider_defaults.get("_metadata", {})
        profile_line = ""
        if profile_metadata.get("profile"):
            success_rate = profile_metadata.get("measured_success_rate")
            if success_rate is None:
                reason = profile_metadata.get("reason", "midpoint defaults")
                profile_line = f"Slider profile=neutral  ({reason})"
            else:
                profile_line = (
                    f"Slider profile={profile_metadata['profile']}  "
                    f"({success_rate:.1%})"
                )
        metrics_text.set_text(
            "\n".join(
                [
                    profile_line,
                    f"Hit={int(final_frame.success)}  Truncated={int(final_frame.truncated)}",
                    f"Best policy accuracy={BEST_POLICY_ACCURACY:.1%}",
                    f"Episode time={data['time'][-1]:.3f}s",
                    f"Total reward={state['reward']:.3f}",
                    f"Click distance={click_distance:.4f} m",
                    f"Click/radius={click_distance / max(final_frame.target_radius, 1e-12):.2f}",
                    f"Target radius={final_frame.target_radius:.4f} m",
                    f"Target speed={slider_value('target_speed'):.3f} m/s",
                    f"Target direction={slider_value('target_direction'):.0f} deg",
                    f"Physical mouse path={mouse_path:.4f} m",
                    f"Final rotation={rotation_degrees[-1]:.1f} deg",
                ]
            )
        )

    def update_frame(frame_index):
        if not state["frames"]:
            return []
        data = state["data"]
        frames = state["frames"]
        index = int(np.clip(frame_index, 0, len(frames) - 1))
        state["frame"] = index
        frame = frames[index]

        cursor_reference_line.set_data(
            data["cursor_reference"][:index + 1, 0], data["cursor_reference"][:index + 1, 1]
        )
        cursor_line.set_data(data["cursor"][:index + 1, 0], data["cursor"][:index + 1, 1])
        target_line.set_data(data["target"][:index + 1, 0], data["target"][:index + 1, 1])
        cursor_point.set_data([frame.cursor_pos[0]], [frame.cursor_pos[1]])
        target_point.set_data([frame.target_pos[0]], [frame.target_pos[1]])
        target_circle.center = frame.target_pos
        target_circle.radius = frame.target_radius
        hand_trail_line.set_data(data["hand"][:index + 1, 0], data["hand"][:index + 1, 1])
        mouse_center.set_data([frame.hand_pos[0]], [frame.hand_pos[1]])

        mouse_transform = (
            Affine2D().rotate(frame.mouse_orientation).translate(*frame.hand_pos) + desk_axis.transData
        )
        mouse_sprite.set_transform(mouse_transform)
        rotation_text.set_text(f"mouse rotation: {np.degrees(frame.mouse_orientation):6.1f} deg")

        if index == len(frames) - 1:
            click_position = frame.click_pos if frame.click_pos is not None else frame.cursor_pos
            click_point.set_data([click_position[0]], [click_position[1]])
        else:
            click_point.set_data([], [])
        distance_cursor.set_xdata([data["time"][index], data["time"][index]])
        rotation_cursor.set_xdata([data["time"][index], data["time"][index]])
        return []

    def tick(_):
        if state["playing"] and state["frames"]:
            last_frame = len(state["frames"]) - 1
            next_frame = min(state["frame"] + 1, last_frame)
            update_frame(next_frame)
            if next_frame >= last_frame:
                state["playing"] = False
                play_button.label.set_text("Play")
                if state["animation"] is not None:
                    state["animation"].event_source.stop()
        return []

    def on_run(_):
        collect_episode()
        play_button.label.set_text("Pause")
        if state["animation"] is not None:
            state["animation"].event_source.start()
        figure.canvas.draw_idle()

    def on_slider_changed(_):
        if state["suppress_slider_events"]:
            return
        on_run(None)

    def on_play(_):
        if state["frame"] >= len(state["frames"]) - 1:
            state["frame"] = 0
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")
        if state["animation"] is not None:
            if state["playing"]:
                state["animation"].event_source.start()
            else:
                state["animation"].event_source.stop()

    def on_reset(_):
        state["suppress_slider_events"] = True
        try:
            for slider in sliders.values():
                slider.reset()
        finally:
            state["suppress_slider_events"] = False
        on_run(None)

    run_button.on_clicked(on_run)
    play_button.on_clicked(on_play)
    reset_button.on_clicked(on_reset)
    slider_callback_ids = [
        slider.on_changed(on_slider_changed)
        for slider in sliders.values()
    ]
    collect_episode()

    live_animation = None
    if interactive:
        interval_ms = max(1, int(1000 / max(fps, 1)))
        live_animation = animation.FuncAnimation(
            figure, tick, interval=interval_ms, blit=False, cache_frame_data=False
        )
        state["animation"] = live_animation
        figure._pnc_animation = live_animation

    # Matplotlib widgets only remain interactive while strong references to
    # them exist. The builder returns before ``plt.show()``, so retain every
    # widget and callback on the figure for the full window lifetime.
    figure._pnc_widgets = {
        "sliders": sliders,
        "buttons": (run_button, play_button, reset_button),
        "callbacks": (
            on_run,
            on_play,
            on_reset,
            on_slider_changed,
        ),
        "slider_callback_ids": slider_callback_ids,
    }

    def make_export_animation():
        state["playing"] = False
        return animation.FuncAnimation(
            figure,
            update_frame,
            frames=len(state["frames"]),
            interval=1000 / max(fps, 1),
            blit=False,
            repeat=False,
        )

    return figure, live_animation, make_export_animation, state


def save_animation(animation, output_path: Path, fps: float, dpi: int):
    from matplotlib.animation import FFMpegWriter, PillowWriter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("FFmpeg is required for MP4 export. Use a .gif output instead.")
        writer = FFMpegWriter(fps=fps, bitrate=2400)
    else:
        raise ValueError("--save must end in .gif or .mp4")
    animation.save(str(output_path), writer=writer, dpi=dpi)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate target/cursor positions on the monitor and hand position on the desk."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run name, run directory, or SAC .zip path; newest run by default",
    )
    parser.add_argument(
        "--ckpt",
        default="last",
        help="Checkpoint for a run name: 'last' (default), 'best', or an integer step count",
    )
    parser.add_argument(
        "--slider-profile",
        choices=("last", "best", "midpoint"),
        default="last",
        help="Initial slider values: latest evaluation profile (default), best profile, or range midpoints",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--env_class_type", default="PnCEnvDefault")
    parser.add_argument("--agent_type", default="classic")
    parser.add_argument("--pnc_type", default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--save", type=Path, default=None, help="Optional .gif or .mp4 output")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the policy mean")
    parser.add_argument("--no-show", action="store_true", help="Do not open the interactive window")
    parser.add_argument("--no-repeat", action="store_true", help="Do not loop the interactive animation")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model, args.ckpt)

    env = build_environment(
        model_path,
        env_class_name=args.env_class_type,
        agent_type=args.agent_type,
        pnc_type=args.pnc_type,
    )
    model = SAC.load(str(model_path), device=args.device)
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(
            f"Model observation shape {model.observation_space.shape} does not match "
            f"environment shape {env.observation_space.shape}."
        )

    slider_defaults = load_slider_profile(model_path, args.slider_profile)
    figure, live_animation, make_export_animation, state = create_parameter_visualizer(
        model,
        env,
        model_path.parent.parent.name,
        seed=args.seed,
        fps=args.fps,
        deterministic=not args.stochastic,
        interactive=not args.no_show,
        slider_defaults=slider_defaults,
    )

    if args.save is not None:
        export_animation = make_export_animation()
        save_animation(export_animation, args.save.resolve(), args.fps, args.dpi)
        print(f"Saved visualization: {args.save.resolve()}")
    frames = state["frames"]
    print(
        f"Model: {model_path}\nEpisode: {len(frames)-1} steps, "
        f"reward={state['reward']:.3f}, success={frames[-1].success}"
    )

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(figure)


if __name__ == "__main__":
    main()
