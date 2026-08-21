import unittest

import numpy as np

from src.utils.raw_count_features import (
    RAW_COUNT_FEATURE_NAMES,
    aggregate_count_events,
    count_event_summary_features,
)


class RawCountFeatureTests(unittest.TestCase):
    def test_events_are_aggregated_with_end_of_bin_semantics(self):
        binned = aggregate_count_events(
            [0.001, 0.008, 0.009, 0.016],
            [[1, 0], [2, 0], [0, 3], [0, 4]],
            movement_start_s=0.0,
            movement_end_s=0.016,
            bin_hz=125,
        )

        np.testing.assert_array_equal(binned, [[3, 0], [0, 7]])

    def test_summary_stays_in_count_and_time_units(self):
        features = count_event_summary_features(
            [0.008, 0.016],
            [[3, 4], [0, 10]],
            movement_start_s=0.0,
            movement_end_s=0.016,
            bin_hz=125,
        )

        self.assertEqual(set(features), set(RAW_COUNT_FEATURE_NAMES))
        self.assertEqual(features["raw_count_moving_bins"], 2.0)
        self.assertEqual(features["raw_count_mean_per_moving_bin"], 7.5)
        self.assertEqual(features["raw_count_max_per_bin"], 10.0)
        self.assertAlmostEqual(features["raw_count_path_per_second"], 937.5)


if __name__ == "__main__":
    unittest.main()
