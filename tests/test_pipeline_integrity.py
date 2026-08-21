import unittest

import numpy as np
import pandas as pd

from src.experiment.task_design import make_trial_stage_template, make_trial_templates
from src.agent.pnc_agent import PnCEnvCpiHidden, PnCEnvDefault
from src.config.constant import FIELD
from src.inference.pnc_dataset import PnCSimulatorForAmortizer
from src.utils.mymath import recovery_metrics
from src.utils.trajectory_preprocessing import (
    endpoint_preserving_grid,
    resample_feature_frame,
)


class RecoveryMetricTests(unittest.TestCase):
    def test_reversed_order_is_not_reported_as_perfect_recovery(self):
        metrics = recovery_metrics([400, 800, 1600], [1600, 800, 400])
        self.assertLess(metrics["pearson_r"], 0.0)
        self.assertLess(metrics["r2"], 0.0)
        self.assertLess(metrics["slope"], 0.0)


class TrajectoryPreprocessingTests(unittest.TestCase):
    def test_grid_uses_exact_rate_and_explicit_click_endpoint(self):
        grid = endpoint_preserving_grid(0.23, 20)
        np.testing.assert_allclose(grid, [0.0, 0.05, 0.10, 0.15, 0.20, 0.23])

    def test_feature_resampling_keeps_click_endpoint(self):
        frame = pd.DataFrame(
            {
                "timestamp": [0.0, 0.11, 0.23],
                "timediff": [0.0, 0.11, 0.12],
                "x": [0.0, 0.5, 1.0],
            }
        )
        result = resample_feature_frame(frame, ["timediff", "x"], 20)
        self.assertAlmostEqual(result[-1, 0], 0.03)
        self.assertAlmostEqual(result[-1, 1], 1.0)


class HumanTaskGeometryTests(unittest.TestCase):
    def test_id_and_radius_ranges_are_preserved(self):
        templates = make_trial_templates(
            1000,
            (0.5313124474, 0.2988632517),
            (2.5, 7.2),
            20260819,
            target_radius_range_m=(0.00125, 0.007),
        )
        self.assertTrue(all(2.5 <= item.fitts_id <= 7.2 for item in templates))
        self.assertTrue(
            all(0.00125 <= item.target_radius_m <= 0.007 for item in templates)
        )

    def test_start_snap_makes_task_geometry_exact(self):
        base = make_trial_templates(
            1,
            (0.5313124474, 0.2988632517),
            (2.5, 7.2),
            1,
            target_radius_range_m=(0.00125, 0.007),
        )[0]
        start = make_trial_stage_template(
            base, "start", np.zeros(2), (0.5313124474, 0.2988632517)
        )
        task = make_trial_stage_template(
            base,
            "task",
            np.array([start.target_x_m, start.target_y_m]),
            (0.5313124474, 0.2988632517),
        )
        self.assertAlmostEqual(task.target_radius_m, base.target_radius_m)
        self.assertAlmostEqual(task.target_distance_m, base.target_distance_m)
        self.assertAlmostEqual(task.fitts_id, base.fitts_id)


class EnvironmentContractTests(unittest.TestCase):
    def test_default_constructors_produce_valid_observations(self):
        for env_class in (PnCEnvDefault, PnCEnvCpiHidden):
            with self.subTest(env=env_class.__name__):
                env = env_class()
                observation, _ = env.reset(seed=7)
                self.assertTrue(env.observation_space.contains(observation))

    def test_hidden_environment_masks_only_cpi_parameter_slot(self):
        visible = PnCEnvDefault()
        hidden = PnCEnvCpiHidden()
        visible.fix_z({"cpi": 1.0})
        hidden.fix_z({"cpi": 1.0})
        visible_obs, _ = visible.reset(seed=11)
        hidden_obs, _ = hidden.reset(seed=11)
        cpi_index = list(visible.user.param_modul.list).index("cpi")
        self.assertEqual(hidden_obs[cpi_index], 0.0)
        np.testing.assert_allclose(
            np.delete(hidden_obs, cpi_index),
            np.delete(visible_obs, cpi_index),
        )


class CompletedClickSamplingTests(unittest.TestCase):
    def test_timeout_is_replaced_by_a_fresh_task(self):
        class FakeGame:
            def __init__(self):
                self.next_task = 0

            def sample_task_conditions(self, count, fix_cond_list=None, **kwargs):
                tasks = [
                    {"task_id": index}
                    for index in range(self.next_task, self.next_task + count)
                ]
                self.next_task += count
                return tasks

        class FakeSimulator:
            def __init__(self):
                self.tasks = []

            def sample_user_stat_conditions(self, count, **kwargs):
                return [{} for _ in range(count)]

            def simulate(self, task_list, user_stat_list, **kwargs):
                self.tasks = list(task_list)

            def get_simulation_result(self, return_traj, downsample):
                # Only the first sampled task times out.  A correct retry must
                # sample task 2 instead of replaying task 0 deterministically.
                truncated = [task["task_id"] == 0 for task in self.tasks]
                stats = pd.DataFrame(
                    {
                        FIELD.TRUNC: truncated,
                        "task_id": [task["task_id"] for task in self.tasks],
                    }
                )
                trajectories = [task["task_id"] for task in self.tasks]
                return stats, trajectories

            def clear_simulation(self):
                self.tasks = []

        sampler = object.__new__(PnCSimulatorForAmortizer)
        sampler.game = FakeGame()
        sampler.simulator = FakeSimulator()
        sampler.fixed_condition_per_param = []
        sampler.task_cond = {}
        sampler.user_stat_cond = {}
        sampler.downsample_rate = 20

        stats, trajectories, attempts = sampler._simulate_completed_trials(2)

        self.assertEqual(attempts, 3)
        self.assertEqual(stats["task_id"].tolist(), [1, 2])
        self.assertEqual(trajectories, [1, 2])


if __name__ == "__main__":
    unittest.main()
