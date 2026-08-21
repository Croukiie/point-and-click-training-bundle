import numpy as np
import unittest

from src.config.config import INF
from src.utils.trajectory_features import (
    CURVATURE_FEATURE_NAMES,
    cursor_curvature_features,
)


def _circular_arc(n_samples: int, radius: float = 1.0) -> np.ndarray:
    angle = np.linspace(0.0, np.pi / 2.0, n_samples)
    return radius * np.column_stack((np.cos(angle), np.sin(angle)))


class TrajectoryCurvatureTests(unittest.TestCase):
    def test_straight_path_has_zero_turn_and_deviation(self):
        path = np.column_stack((np.linspace(0.0, 1.0, 100), np.zeros(100)))
        features = cursor_curvature_features(path, target_position=[1.0, 0.0])

        self.assertEqual(set(features), set(CURVATURE_FEATURE_NAMES))
        for value in features.values():
            self.assertTrue(np.isfinite(value))
            self.assertAlmostEqual(value, 0.0, delta=1e-10)

    def test_curved_path_has_expected_geometric_signal(self):
        path = _circular_arc(200)
        features = cursor_curvature_features(path, target_position=path[-1])

        self.assertAlmostEqual(
            features["cursor_total_abs_turn_rad"], np.pi / 2.0, delta=0.08
        )
        self.assertAlmostEqual(
            features["cursor_net_turn_rad"], np.pi / 2.0, delta=0.08
        )
        self.assertGreater(features["cursor_max_deviation_normalized"], 0.1)
        self.assertGreater(features["cursor_path_excess_ratio"], 0.0)

    def test_features_are_stable_across_scale_rotation_and_sampling_rate(self):
        sparse = _circular_arc(24)
        dense = _circular_arc(500)
        angle = 0.73
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        transformed = (dense @ rotation.T) * 4.7 + np.array([12.0, -3.0])

        sparse_features = cursor_curvature_features(
            sparse, target_position=sparse[-1]
        )
        transformed_features = cursor_curvature_features(
            transformed, target_position=transformed[-1]
        )

        for name in CURVATURE_FEATURE_NAMES:
            self.assertAlmostEqual(
                transformed_features[name],
                sparse_features[name],
                delta=max(0.015, abs(sparse_features[name]) * 0.04),
            )

    def test_duplicate_and_non_finite_samples_do_not_break_features(self):
        path = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [np.nan, 0.0],
                [0.5, 0.1],
                [0.5, 0.1],
                [1.0, 0.0],
            ]
        )
        features = cursor_curvature_features(path, target_position=[1.0, 0.0])
        self.assertTrue(all(np.isfinite(value) for value in features.values()))

    def test_extractor_has_no_cpi_or_gain_argument(self):
        argument_names = cursor_curvature_features.__code__.co_varnames[
            : cursor_curvature_features.__code__.co_argcount
        ]
        self.assertNotIn("cpi", argument_names)
        self.assertNotIn("mouse_gain", argument_names)

    def test_curvature_amortizer_profile_contains_no_cpi_derived_hand_feature(self):
        config = INF.normalize.curvature
        exposed_features = list(config.stat.list) + list(config.traj.list)
        forbidden_fragments = ("cpi", "gain", "hand", "raw_")

        self.assertTrue(
            set(CURVATURE_FEATURE_NAMES).issubset(set(config.stat.list))
        )
        self.assertFalse(
            any(
                fragment in feature
                for feature in exposed_features
                for fragment in forbidden_fragments
            )
        )


if __name__ == "__main__":
    unittest.main()
