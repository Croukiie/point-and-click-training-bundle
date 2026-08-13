"""Portable contracts adapted from the colleague test organization."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from src.experiment.export_human_amortizer_data import _resample_movement
except ModuleNotFoundError:
    _resample_movement = None
from src.scripts.shared_cpi_ablation_matrix import (
    ABLATION_ORDER,
    PROFILE_REMOVED_FEATURES,
    matrix_manifest,
    register_profiles,
    retained_curvature,
)
from src.scripts.shared_cpi_2step_run import (
    BLOCK_PARAMETERS,
    CURVATURE_FEATURES,
    SHARED_PARAMETERS,
    build_data_config,
    build_simulator_config,
)
from src.utils.trajectory_features import cursor_curvature_features


class AblationMatrixContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_profiles()

    def test_profiles_form_the_intended_zero_three_seven_ten_ladder(self):
        expected = {
            "speedpath": 0,
            "core3": 3,
            "clean7": 7,
            "full10": 10,
        }
        self.assertEqual(tuple(ABLATION_ORDER), tuple(expected))
        for profile, count in expected.items():
            with self.subTest(profile=profile):
                self.assertEqual(len(retained_curvature(profile)), count)
                self.assertEqual(
                    set(retained_curvature(profile))
                    | set(PROFILE_REMOVED_FEATURES[profile]),
                    set(CURVATURE_FEATURES),
                )

    def test_profiles_change_only_curvature_summary_inputs(self):
        base_features = None
        for profile in ABLATION_ORDER:
            config = build_data_config("summary-only", profile)
            self.assertTrue(bool(config.traj.ignore_traj))
            current_base = [
                feature
                for feature in config.stat.list
                if feature not in CURVATURE_FEATURES
            ]
            if base_features is None:
                base_features = current_base
            self.assertEqual(current_base, base_features)
            self.assertEqual(
                [
                    feature
                    for feature in config.stat.list
                    if feature in CURVATURE_FEATURES
                ],
                list(retained_curvature(profile)),
            )

    def test_shared_parameter_partition_is_unchanged(self):
        simulator = build_simulator_config("model", "best", 20)
        self.assertEqual(tuple(simulator.main_param), SHARED_PARAMETERS)
        self.assertEqual(tuple(simulator.sub_param), BLOCK_PARAMETERS)
        self.assertEqual(len(simulator.main_param), 9)
        self.assertEqual(tuple(simulator.sub_param), ("cpi",))
        self.assertNotIn("cpi", simulator.main_param)

    def test_manifest_has_stable_array_indices(self):
        profiles = matrix_manifest()["profiles"]
        self.assertEqual([item["index"] for item in profiles], [0, 1, 2, 3])
        self.assertEqual([item["name"] for item in profiles], list(ABLATION_ORDER))


class ImportedCurvatureContracts(unittest.TestCase):
    def test_early_and_late_turn_features_distinguish_path_phases(self):
        x = np.linspace(0.0, 1.0, 1001)
        y = np.where(x <= 0.55, 0.0, 2.0 * (x - 0.55) ** 2)
        features = cursor_curvature_features(
            np.column_stack((x, y)),
            target_position=(1.0, y[-1]),
        )
        self.assertGreater(
            features["cursor_late_mean_abs_turn_rad"],
            features["cursor_early_mean_abs_turn_rad"] + 1e-4,
        )


@unittest.skipIf(
    _resample_movement is None,
    "human export module is not installed in this bundle",
)
class ImportedHumanResamplingContracts(unittest.TestCase):
    def test_duplicate_timestamps_keep_the_last_cursor_position(self):
        raw = [
            {"time_from_onset_ns": "0", "cursor_x_m": "0", "cursor_y_m": "0"},
            {"time_from_onset_ns": "500000000", "cursor_x_m": "0.2", "cursor_y_m": "0"},
            {"time_from_onset_ns": "500000000", "cursor_x_m": "0.5", "cursor_y_m": "0"},
            {"time_from_onset_ns": "1000000000", "cursor_x_m": "1", "cursor_y_m": "0"},
        ]
        trial = {
            "reaction_time_s": "0",
            "target_x_m": "1",
            "target_y_m": "0",
        }
        trajectory, max_speed = _resample_movement(raw, trial, 2.0)
        self.assertAlmostEqual(trajectory[1]["cursor_pos_x"], 0.5)
        self.assertAlmostEqual(trajectory[-1]["cursor_pos_x"], 1.0)
        self.assertAlmostEqual(max_speed, 1.0)

    def test_resampling_preserves_the_terminal_sample(self):
        raw = [
            {"time_from_onset_ns": "100000000", "cursor_x_m": "0", "cursor_y_m": "0"},
            {"time_from_onset_ns": "330000000", "cursor_x_m": "0.2", "cursor_y_m": "0.1"},
        ]
        trial = {
            "reaction_time_s": "0.1",
            "target_x_m": "0.2",
            "target_y_m": "0.1",
        }
        trajectory, _ = _resample_movement(raw, trial, 20.0)
        self.assertAlmostEqual(trajectory[0]["timestamp"], 0.0)
        self.assertAlmostEqual(trajectory[-1]["timestamp"], 0.23)
        self.assertAlmostEqual(trajectory[-1]["cursor_pos_x"], 0.2)
        self.assertAlmostEqual(trajectory[-1]["cursor_pos_y"], 0.1)

    def test_stationary_resampled_trial_has_zero_speed(self):
        raw = [
            {"time_from_onset_ns": "0", "cursor_x_m": "0.2", "cursor_y_m": "-0.1"},
            {"time_from_onset_ns": "1000000000", "cursor_x_m": "0.2", "cursor_y_m": "-0.1"},
        ]
        trial = {
            "reaction_time_s": "0",
            "target_x_m": "0.2",
            "target_y_m": "-0.1",
        }
        _, max_speed = _resample_movement(raw, trial, 20.0)
        self.assertEqual(max_speed, 0.0)


if __name__ == "__main__":
    unittest.main()
