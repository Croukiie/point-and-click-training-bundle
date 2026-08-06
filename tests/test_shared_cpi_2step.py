import unittest

import numpy as np
import torch

from src.experiment.infer_human_shared_cpi import parse_trial_counts
from src.nets.amortizer import ConditionalRegressionForTrialData
from src.scripts.shared_cpi_2step_run import (
    BLOCK_PARAMETERS,
    SHARED_PARAMETERS,
    build_data_config,
    build_simulator_config,
    build_training_config,
)


class SharedCpiConfigurationTests(unittest.TestCase):
    def test_parameter_partition_is_nine_shared_plus_cpi(self):
        simulator = build_simulator_config("rl_model", "best", 20)
        self.assertEqual(tuple(simulator.main_param), SHARED_PARAMETERS)
        self.assertEqual(tuple(simulator.sub_param), BLOCK_PARAMETERS)
        self.assertEqual(set(simulator.main_param) & set(simulator.sub_param), set())
        self.assertEqual(len(simulator.main_param), 9)
        self.assertEqual(simulator.sub_param, ["cpi"])

    def test_variants_only_change_trajectory_usage(self):
        trajectory = build_data_config("trajectory")
        summary = build_data_config("summary-only")
        self.assertFalse(trajectory.traj.ignore_traj)
        self.assertTrue(summary.traj.ignore_traj)
        self.assertEqual(trajectory.stat.list, summary.stat.list)

    def test_trial_count_parser(self):
        self.assertIsNone(parse_trial_counts("all"))
        self.assertEqual(parse_trial_counts("64,16,64"), [16, 64])


class ConditionalTrajectoryRegressionTests(unittest.TestCase):
    def test_forward_and_infer_support_three_trajectory_blocks(self):
        config = build_training_config("cpu").amortizer
        config.encoder.stat_sz = 2
        config.encoder.traj_sz = 2
        config.encoder.transformer.max_step = 12
        config.linear_main.out_sz = len(SHARED_PARAMETERS)
        config.linear_main.in_sz = config.trial_encoder.attention.out_sz
        config.linear_sub.out_sz = len(BLOCK_PARAMETERS)
        config.linear_sub.in_sz = (
            config.trial_encoder.attention.out_sz + len(SHARED_PARAMETERS)
        )
        model = ConditionalRegressionForTrialData(config)

        batch_stats = np.zeros((2, 3, 4, 2), dtype=np.float32)
        batch_trajectories = np.empty((2, 3, 4), dtype=object)
        for index in np.ndindex(batch_trajectories.shape):
            batch_trajectories[index] = np.zeros((5, 2), dtype=np.float32)

        output = model(batch_stats, batch_trajectories)
        self.assertEqual(tuple(output.shape), (2, 12))

        flat_stats = batch_stats[0].reshape(12, 2)
        flat_trajectories = list(batch_trajectories[0].reshape(-1))
        grouping = torch.arange(3).repeat_interleave(4)
        main, sub = model.infer(
            flat_stats,
            flat_trajectories,
            grouping_index=grouping,
        )
        self.assertEqual(main.shape, (9,))
        self.assertEqual(sub.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
