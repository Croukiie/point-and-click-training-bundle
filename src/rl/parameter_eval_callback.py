"""Evaluation callback that persists the parameter profile of every episode."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

from ..utils.myutils import save_dict_to_yaml


class ParameterTrackingEvalCallback(EvalCallback):
    """EvalCallback with per-episode parameter logging and visualizer profile export."""

    def __init__(self, *args, parameter_log_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter_log_path = Path(parameter_log_path)
        self._evaluation_index = 0
        self._current_episode_records = []

    def _init_callback(self) -> None:
        super()._init_callback()
        self.parameter_log_path.mkdir(parents=True, exist_ok=True)

    def _log_success_callback(self, locals_, globals_) -> None:
        super()._log_success_callback(locals_, globals_)
        if not locals_["done"]:
            return

        info = locals_["info"]
        parameters = info.get("episode_parameters")
        if parameters is None:
            return

        env_index = int(locals_["i"])
        self._current_episode_records.append(
            {
                "timesteps": int(self.num_timesteps),
                "evaluation_index": int(self._evaluation_index + 1),
                "episode_index": len(self._current_episode_records),
                "reward": float(locals_["current_rewards"][env_index]),
                "length": int(locals_["current_lengths"][env_index]),
                "success": bool(info.get("is_success", False)),
                "parameters": parameters,
            }
        )

    @staticmethod
    def _append_jsonl(path: Path, records) -> None:
        with path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_evaluation_logs(self, is_new_best: bool) -> None:
        records = self._current_episode_records
        if not records:
            return

        self._append_jsonl(self.parameter_log_path / "episodes.jsonl", records)
        rewards = np.asarray([record["reward"] for record in records], dtype=float)
        successes = np.asarray([record["success"] for record in records], dtype=float)
        lengths = np.asarray([record["length"] for record in records], dtype=float)
        summary = {
            "timesteps": int(self.num_timesteps),
            "evaluation_index": int(self._evaluation_index),
            "episodes": len(records),
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "success_rate": float(successes.mean()),
            "mean_episode_length": float(lengths.mean()),
            "new_best_mean_reward": bool(is_new_best),
        }
        self._append_jsonl(self.parameter_log_path / "summaries.jsonl", [summary])

        self._write_last_profile(summary, records)
        if is_new_best:
            self._write_best_profile(summary, records)

    @staticmethod
    def _select_display_episode(records):
        successful_records = [record for record in records if record["success"]]
        candidates = successful_records or records

        def rank(record):
            outcome = record["parameters"].get("outcome", {})
            return (
                record["reward"],
                -float(outcome.get("click_distance", np.inf)),
                -record["length"],
            )

        return max(candidates, key=rank)

    def _make_visualizer_profile(self, summary, selected_episode, *, profile, checkpoint, criterion):
        metadata = selected_episode["parameters"]
        task = dict(metadata.get("task", {}))
        return {
            "profile": profile,
            "checkpoint": checkpoint,
            "criterion": criterion,
            "evaluation_timesteps": int(self.num_timesteps),
            "evaluation_episodes": int(summary["episodes"]),
            "evaluation_mean_reward": float(summary["mean_reward"]),
            "measured_success_rate": float(summary["success_rate"]),
            "selected_episode_reward": float(selected_episode["reward"]),
            "selected_episode_success": bool(selected_episode["success"]),
            "parameters": dict(metadata.get("actual", {})),
            "normalized_parameters": dict(metadata.get("normalized", {})),
            "task": {
                "target_radius": task.get("target_radius"),
                "target_speed": task.get("target_speed"),
                "target_direction": task.get("target_direction"),
                "target_difficulty": task.get("target_difficulty"),
            },
            "outcome": dict(metadata.get("outcome", {})),
        }

    def _write_last_profile(self, summary, records) -> None:
        selected_episode = self._select_display_episode(records)
        profile = self._make_visualizer_profile(
            summary,
            selected_episode,
            profile="last_evaluation_episode",
            checkpoint="last",
            criterion="latest_evaluation_then_successful_episode_reward",
        )

        run_path = Path(self.best_model_save_path).parent
        save_dict_to_yaml(profile, str(run_path / "visualizer_last_params.yaml"))
        save_dict_to_yaml(
            {**summary, "selected_profile": profile},
            str(run_path / "last_eval_summary.yaml"),
        )
        with (run_path / "last_eval_episodes.json").open("w", encoding="utf-8") as stream:
            json.dump(records, stream, ensure_ascii=False, indent=2)

    def _write_best_profile(self, summary, records) -> None:
        best_episode = self._select_display_episode(records)
        profile = self._make_visualizer_profile(
            summary,
            best_episode,
            profile="best_evaluation_episode",
            checkpoint="best",
            criterion="best_mean_reward_then_successful_episode_reward",
        )

        best_path = Path(self.best_model_save_path)
        save_dict_to_yaml(profile, str(best_path / "visualizer_best_params.yaml"))
        save_dict_to_yaml(
            {**summary, "selected_profile": profile},
            str(best_path / "best_eval_summary.yaml"),
        )
        with (best_path / "best_eval_episodes.json").open("w", encoding="utf-8") as stream:
            json.dump(records, stream, ensure_ascii=False, indent=2)

    def _on_step(self) -> bool:
        will_evaluate = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        previous_best = self.best_mean_reward
        if will_evaluate:
            self._current_episode_records = []

        continue_training = super()._on_step()

        if will_evaluate:
            self._evaluation_index += 1
            is_new_best = self.best_mean_reward > previous_best
            self._write_evaluation_logs(is_new_best)
        return continue_training
