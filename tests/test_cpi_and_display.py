import math
import unittest
from copy import deepcopy

import numpy as np

from src.agent.pnc_agent import PnCEnvDefault
from src.agent.pnc_task import PnCGame
from src.config.config import ENV, INF
from src.config.constant import FIELD
from src.scripts.all_in_one_run import build_parser
from src.utils.mouse_sensitivity import (
    cursor_meters_per_count,
    effective_cursor_gain,
    physical_meters_per_count,
)
from src.utils.mymath import linear_denormalize, linear_normalize
from src.utils.myutils import mask_and_pad_traj_data


class CpiAndDisplayTests(unittest.TestCase):
    def test_24_inch_16_by_9_geometry(self):
        width = ENV.default.range.window_width.max
        height = width * ENV.window_ratio_height / ENV.window_ratio_width
        diagonal_inches = math.hypot(width, height) / 0.0254

        self.assertAlmostEqual(diagonal_inches, 24.0, places=6)
        self.assertAlmostEqual(width, 0.5313124474, places=9)
        self.assertAlmostEqual(height, 0.2988632517, places=9)

    def test_initial_mouse_gain_matches_full_hd_pixel_pitch(self):
        width = ENV.default.range.window_width.max
        reference_cpi = ENV.default.mouse_reference_cpi
        expected_gain = width * reference_cpi / (0.0254 * 1920)

        self.assertAlmostEqual(ENV.default.mouse_gain, expected_gain, places=14)
        self.assertAlmostEqual(
            cursor_meters_per_count(ENV.default.mouse_gain, reference_cpi),
            width / 1920,
            places=15,
        )

    def test_cpi_scales_cursor_motion_linearly(self):
        game = PnCGame("default")
        game.mouse_rotation_gain_x = 0.0
        game.mouse_rotation_gain_y = 0.0
        hand_start = np.array([0.0, 0.0])
        hand_end = np.array([17.0, -8.0]) * physical_meters_per_count(400)

        game.cpi = 400
        delta_400, _ = game._cursor_delta_from_hand_segment(
            hand_start, hand_end, mouse_ori=0.0
        )
        game.cpi = 1600
        delta_1600, _ = game._cursor_delta_from_hand_segment(
            hand_start, hand_end, mouse_ori=0.0
        )

        np.testing.assert_allclose(delta_1600, 4.0 * delta_400)
        self.assertAlmostEqual(game.cursor_gain, 4.0 * game.mouse_gain)

    def test_sub_count_residual_is_carried_between_segments(self):
        game = PnCGame("default")
        game.cpi = 400
        game.mouse_rotation_gain_x = 0.0
        game.mouse_rotation_gain_y = 0.0
        segment = np.array([0.4, 0.0]) * physical_meters_per_count(400)

        first_delta, _, residual = game._cursor_delta_from_hand_segment(
            np.zeros(2),
            segment,
            mouse_ori=0.0,
            count_residual=np.zeros(2),
            return_count_residual=True,
        )
        second_delta, _, residual = game._cursor_delta_from_hand_segment(
            segment,
            2 * segment,
            mouse_ori=0.0,
            count_residual=residual,
            return_count_residual=True,
        )

        np.testing.assert_allclose(first_delta, np.zeros(2))
        self.assertAlmostEqual(
            second_delta[0],
            cursor_meters_per_count(game.mouse_gain, 400),
        )
        np.testing.assert_allclose(residual, [-0.2, 0.0], atol=1e-12)

    def test_sensor_count_replay_uses_identical_residual_carry(self):
        game = PnCGame("default")
        game.cpi = 400
        game.mouse_mapping = "linear"
        count_size = physical_meters_per_count(400)
        hand_positions = np.array(
            [[0.0, 0.0], [0.4 * count_size, 0.0], [0.8 * count_size, 0.0]]
        )

        counts = game.get_sensor_count_deltas(hand_positions)

        np.testing.assert_array_equal(counts, [[0, 0], [1, 0]])

    def test_virtual_rollout_does_not_commit_sensor_residual(self):
        game = PnCGame("default")
        game.cpi = 400
        game.mouse_rotation_gain_x = 0.0
        game.mouse_rotation_gain_y = 0.0
        segment = np.array([0.4, 0.0]) * physical_meters_per_count(400)
        hand_positions = np.array([[0.0, 0.0], segment])
        hand_velocities = np.zeros_like(hand_positions)

        game.get_cursor_and_hand_update(hand_positions, hand_velocities)
        np.testing.assert_allclose(game.mouse_count_residual, np.zeros(2))

        game.update_cursor_and_hand_move(hand_positions, hand_velocities)
        np.testing.assert_allclose(
            game.mouse_count_residual,
            [0.4, 0.0],
            atol=1e-12,
        )

    def test_same_cursor_plan_requires_less_hand_motion_at_high_cpi(self):
        game = PnCGame("default")
        cursor_positions = np.array([[0.0, 0.0], [0.04, 0.01]])
        cursor_velocities = np.zeros_like(cursor_positions)

        game.cpi = 400
        hand_400, _ = game.convert_cursor_plan_to_hand_plan(
            cursor_positions,
            cursor_velocities,
        )
        game.cpi = 1600
        hand_1600, _ = game.convert_cursor_plan_to_hand_plan(
            cursor_positions,
            cursor_velocities,
        )

        distance_400 = np.linalg.norm(hand_400[-1] - hand_400[0])
        distance_1600 = np.linalg.norm(hand_1600[-1] - hand_1600[0])
        self.assertAlmostEqual(distance_1600, distance_400 / 4)

    def test_cpi_sampling_and_application_use_ten_cpi_steps(self):
        env = PnCEnvDefault(agent_name="classic", pnc_env_name="default")
        cpi_index = env.user.param_modul.list.index("cpi")
        cpi_range = env.user.param_modul.cpi

        samples = np.array([env._z_sampler()[cpi_index] for _ in range(200)])
        values = linear_denormalize(samples, cpi_range.min, cpi_range.max)
        steps = (values - cpi_range.min) / cpi_range.step
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-5)

        env.fix_z(
            {
                "cpi": linear_normalize(
                    735,
                    cpi_range.min,
                    cpi_range.max,
                )
            }
        )
        self.assertEqual(env.user_param["cpi"], 740)
        self.assertEqual(env.pnc_env.cpi, 740)

    def test_initial_orientation_is_random_but_can_be_replayed(self):
        game = PnCGame("default")
        sampled = np.array(
            [game.reset(return_reset_condition=True).mouse_ori for _ in range(1000)]
        )

        self.assertGreater(sampled.std(), np.deg2rad(2))
        self.assertGreaterEqual(sampled.min(), np.deg2rad(-15))
        self.assertLessEqual(sampled.max(), np.deg2rad(15))
        self.assertLess(sampled.min(), np.deg2rad(-13))
        self.assertGreater(sampled.max(), np.deg2rad(13))

        condition = game.reset(mouse_ori=0.123, return_reset_condition=True)
        self.assertAlmostEqual(condition.mouse_ori, 0.123)
        self.assertAlmostEqual(game.mouse_ori, 0.123)

    def test_initial_orientation_fallback_matches_uniform_training_range(self):
        config_without_orientation = deepcopy(ENV.default)
        del config_without_orientation["initial_mouse_orientation"]
        game = PnCGame(config=config_without_orientation)
        sampled = np.degrees(
            [game.reset(return_reset_condition=True).mouse_ori for _ in range(1000)]
        )

        self.assertGreaterEqual(sampled.min(), -15)
        self.assertLessEqual(sampled.max(), 15)
        self.assertLess(sampled.min(), -13)
        self.assertGreater(sampled.max(), 13)

    def test_control_orientation_configs_are_available(self):
        zero_game = PnCGame("default_ori0")
        zero_samples = [
            zero_game.reset(return_reset_condition=True).mouse_ori
            for _ in range(10)
        ]
        np.testing.assert_allclose(zero_samples, np.zeros(10))

        asym_game = PnCGame("default_ori_asym")
        asym_samples = np.degrees(
            [
                asym_game.reset(return_reset_condition=True).mouse_ori
                for _ in range(1000)
            ]
        )
        self.assertGreaterEqual(asym_samples.min(), -10)
        self.assertLessEqual(asym_samples.max(), 20)
        self.assertLess(asym_samples.min(), -8)
        self.assertGreater(asym_samples.max(), 18)

    def test_cpi_cursor_velocity_observation_range_is_wide_enough(self):
        env = PnCEnvDefault(agent_name="classic", pnc_env_name="default")
        self.assertEqual(
            list(env.user.observation.cursor_vel.max),
            [20.0, 20.0],
        )

    def test_experiment_count_conversion_matches_rl_gain(self):
        mouse_gain = ENV.default.mouse_gain
        per_count = cursor_meters_per_count(mouse_gain, 400)
        hand_distance = 0.01
        counts_400 = hand_distance / physical_meters_per_count(400)
        counts_1600 = hand_distance / physical_meters_per_count(1600)

        self.assertAlmostEqual(counts_400 * per_count, hand_distance * mouse_gain)
        self.assertAlmostEqual(
            counts_1600 * per_count,
            hand_distance * effective_cursor_gain(1600, mouse_gain, 400),
        )

    def test_inference_uses_cursor_and_hand_trajectories(self):
        trajectory_config = INF.normalize.default.traj

        self.assertFalse(trajectory_config.ignore_traj)
        self.assertEqual(
            trajectory_config.list,
            [
                FIELD.TIMEDIFF,
                FIELD.CURSOR_POS_X,
                FIELD.CURSOR_POS_Y,
                FIELD.HAND_POS_X,
                FIELD.HAND_POS_Y,
                FIELD.TARGET_POS_X_REL,
                FIELD.TARGET_POS_Y_REL,
            ],
        )

    def test_object_trajectory_arrays_are_padded_as_float32(self):
        trajectories = [
            np.array([[0.0] * 7, [1.0] * 7], dtype=object),
            np.array([[2.0] * 7], dtype=object),
        ]

        padded, mask = mask_and_pad_traj_data(trajectories)

        self.assertEqual(str(padded.dtype), "torch.float32")
        self.assertEqual(tuple(padded.shape), (2, 2, 7))
        self.assertEqual(mask.tolist(), [[True, True], [True, False]])

    def test_all_in_one_generates_two_to_the_seventeenth_parameter_sets(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.train_params, 2**17)


if __name__ == "__main__":
    unittest.main()
