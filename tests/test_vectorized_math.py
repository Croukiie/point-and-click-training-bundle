import unittest

import numpy as np

from src.agent.module_motor import Motor
from src.agent.module_mouse import Mouse
from src.agent.pnc_task import PnCGame
from src.utils.otg import GRAMS, otg_1d, otg_2d


def reference_otg_1d(p0, v0, p1, v1, time_unit, plan_duration, execute_duration):
    gs, _, grams, gram_invs = GRAMS[time_unit]
    plan_steps = plan_duration // time_unit
    execute_steps = execute_duration // time_unit
    x0 = np.array([[p0, v0]]).T
    x1 = np.array([[p1, v1]]).T
    span = gram_invs[plan_steps] @ (x1 - gs[plan_steps] @ x0)
    states = [
        gs[k] @ x0 + grams[plan_steps][k] @ span
        for k in range(execute_steps + 1)
    ]
    states = np.concatenate(states, axis=1)
    return states[0], states[1]


def reference_add_motor_noise(p0, velocity, interval, n_v, n_p):
    noisy_velocity = velocity.copy()
    noise_coefficients = np.array([n_v, n_p])
    for i, current_velocity in enumerate(velocity[1:]):
        speed = np.linalg.norm(current_velocity)
        direction = np.zeros(2) if speed == 0 else current_velocity / speed
        perpendicular = np.array([-direction[1], direction[0]])
        noise = noise_coefficients * speed * np.random.normal(0, 1, 2)
        noisy_velocity[i + 1] += noise[0] * direction + noise[1] * perpendicular

    noisy_position = p0 + np.cumsum(
        (noisy_velocity[1:] + noisy_velocity[:-1]) / 2 * interval / 1000,
        axis=0,
    )
    noisy_position = np.insert(noisy_position, 0, p0, axis=0)
    return noisy_position, noisy_velocity


def reference_endpoint_correction(game, cursor_plan, hand_pos, mouse_orientation):
    intended_delta = cursor_plan[-1] - cursor_plan[0]
    hand_delta_plan = (cursor_plan - cursor_plan[0]) / game.mouse_gain

    def endpoint_error(angle):
        hand_plan = np.array(
            [
                hand_pos + Mouse.rot_mat(delta, mouse_orientation + angle)
                for delta in hand_delta_plan
            ]
        )
        orientation = mouse_orientation
        actual_delta = np.zeros(2)
        for start, end in zip(hand_plan[:-1], hand_plan[1:]):
            cursor_delta, orientation = Mouse.get_cursor_displacement_from_desk_rotation(
                start,
                end,
                game.mouse_gain,
                orientation,
                game.mouse_rotation_gain_x,
                game.mouse_rotation_gain_y,
            )
            actual_delta += cursor_delta
        return np.linalg.norm(actual_delta - intended_delta)

    best_angle = 0.0
    search_width = np.pi / 4
    for _ in range(4):
        candidates = np.linspace(best_angle - search_width, best_angle + search_width, 15)
        best_angle = candidates[np.argmin([endpoint_error(a) for a in candidates])]
        search_width /= 4
    return best_angle


class VectorizedMathTests(unittest.TestCase):
    def test_otg_matches_reference(self):
        cases = [
            (0.1, -0.2, 0.8, 0.3, 50, 500, 100),
            (-0.3, 0.7, 0.2, -0.1, 10, 200, 150),
            (0.0, 0.0, 1.0, 0.0, 1, 50, 30),
        ]
        for case in cases:
            expected_p, expected_v = reference_otg_1d(*case)
            actual_p, actual_v = otg_1d(*case)
            np.testing.assert_allclose(actual_p, expected_p, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(actual_v, expected_v, rtol=1e-13, atol=1e-13)

    def test_batched_otg_matches_individual_trajectories(self):
        p0 = np.array([[0.0, 0.1], [0.2, -0.3], [-0.4, 0.2]])
        v0 = np.array([[0.1, 0.0], [0.0, 0.2], [-0.1, 0.3]])
        p1 = np.array([[0.5, 0.6], [-0.2, 0.4], [0.7, -0.8]])
        v1 = np.zeros_like(v0)
        batch_p, batch_v = otg_2d(p0, v0, p1, v1, 50, 500, 500)
        for i in range(len(p0)):
            scalar_p, scalar_v = otg_2d(p0[i], v0[i], p1[i], v1[i], 50, 500, 500)
            np.testing.assert_allclose(batch_p[:, i], scalar_p, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(batch_v[:, i], scalar_v, rtol=1e-13, atol=1e-13)

    def test_motor_noise_matches_reference_with_same_rng_stream(self):
        velocity = np.array(
            [[0.0, 0.0], [0.4, -0.2], [0.0, 0.0], [-0.3, 0.8], [0.1, 0.2]]
        )
        np.random.seed(1234)
        expected = reference_add_motor_noise(np.array([0.2, -0.1]), velocity, 50, 0.3, 0.1)
        np.random.seed(1234)
        actual = Motor.add_motor_noise(np.array([0.2, -0.1]), velocity, 50, 0.3, 0.1)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-14, atol=1e-14)

    def test_vectorized_mouse_mapping_matches_segment_loop(self):
        rng = np.random.default_rng(7)
        hand_deltas = rng.normal(0, 0.02, size=(12, 2))
        expected_deltas = []
        expected_orientations = []
        orientation = 0.37
        hand_position = np.array([0.1, -0.2])
        for delta in hand_deltas:
            cursor_delta, orientation = Mouse.get_cursor_displacement_from_desk_rotation(
                hand_position, hand_position + delta, 4.16, orientation, -3.0, 1.2
            )
            expected_deltas.append(cursor_delta)
            expected_orientations.append(orientation)
            hand_position += delta

        actual_deltas, actual_orientations = (
            Mouse.get_cursor_displacements_from_desk_rotation(
                hand_deltas, 4.16, 0.37, -3.0, 1.2
            )
        )
        np.testing.assert_allclose(actual_deltas, expected_deltas, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            actual_orientations, expected_orientations, rtol=1e-13, atol=1e-13
        )

    def test_endpoint_correction_matches_reference_search(self):
        game = PnCGame(config_name="default")
        game.mouse_rotation_gain_x = 6.0
        game.mouse_rotation_gain_y = -2.0
        cursor_plan, _ = otg_2d(
            np.array([-0.15, 0.08]),
            np.array([0.05, -0.02]),
            np.array([0.18, -0.12]),
            np.zeros(2),
            50,
            800,
            800,
        )
        expected = reference_endpoint_correction(
            game, cursor_plan, np.array([0.02, -0.01]), 0.31
        )
        actual = game._hand_plan_endpoint_correction(
            cursor_plan, np.array([0.02, -0.01]), 0.31
        )
        self.assertAlmostEqual(actual, expected, places=14)

    def test_interpolation_matches_segment_loop(self):
        trajectory_p, trajectory_v = otg_2d(
            np.array([0.0, 0.1]),
            np.array([0.2, 0.0]),
            np.array([0.5, -0.2]),
            np.zeros(2),
            50,
            500,
            500,
        )
        expected_p = [trajectory_p[:1]]
        expected_v = [trajectory_v[:1]]
        for i in range(len(trajectory_p) - 1):
            segment_p, segment_v = Motor.interp_segment(
                trajectory_p[i],
                trajectory_v[i],
                trajectory_p[i + 1],
                trajectory_v[i + 1],
                50,
                1,
            )
            expected_p.append(segment_p[1:])
            expected_v.append(segment_v[1:])

        actual_p, actual_v = Motor.interpolate_plan(trajectory_p, trajectory_v, 50, 1)
        np.testing.assert_allclose(actual_p, np.concatenate(expected_p), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(actual_v, np.concatenate(expected_v), rtol=1e-11, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
