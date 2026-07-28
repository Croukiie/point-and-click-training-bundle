"""Run CPI-aware RL training, dataset synthesis, and amortizer training.

Run from the repository root:

    python -m src.scripts.all_in_one_run

Individual stages can be reused with ``--skip-rl``, ``--skip-data``, or
``--skip-amortizer``.  When RL is skipped, pass the existing run directory
name through ``--rl-model``.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from box import Box

from ..inference.pnc_dataset import (
    PnCAmortizerTrainingDataset,
    PnCAmortizerValidationDataset,
)
from ..inference.pnc_infer_trainer import PnCAmortizerTrainer
from ..rl.sac_trainer import SACTrainer
from ..utils.myutils import get_current_time_digit


ROOT = Path(__file__).resolve().parents[2]


def parse_checkpoint(value: str):
    if value in {"best", "last"}:
        return value
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "checkpoint must be 'best', 'last', or an integer training step"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the CPI-aware policy, synthesize data, and train its amortizer."
    )
    stages = parser.add_argument_group("pipeline stages")
    stages.add_argument("--skip-rl", action="store_true")
    stages.add_argument("--skip-data", action="store_true")
    stages.add_argument("--skip-amortizer", action="store_true")

    rl = parser.add_argument_group("reinforcement learning")
    rl.add_argument("--rl-model", default=None, help="Existing run name when --skip-rl is used")
    rl.add_argument("--rl-checkpoint", type=parse_checkpoint, default="best")
    rl.add_argument("--rl-cpu", type=int, default=8)
    rl.add_argument("--rl-name", default="pnc_cpi_24in")
    rl.add_argument("--rl-env-class", default="PnCEnvDefault")
    rl.add_argument("--rl-agent", default="classic")
    rl.add_argument("--rl-task", default="default")
    rl.add_argument("--rl-config", default="simple")
    rl.add_argument("--rl-steps", type=int, default=None)
    rl.add_argument("--device", default=None)

    data = parser.add_argument_group("simulation dataset")
    data.add_argument("--data-cpu", type=int, default=8)
    data.add_argument("--train-params", type=int, default=2**17)
    data.add_argument("--trials-per-param", type=int, default=64)
    data.add_argument("--data-repeat", type=int, default=1)
    data.add_argument("--validation-users", type=int, default=100)
    data.add_argument("--validation-trials", type=int, default=256)
    data.add_argument("--trajectory-downsample", type=int, default=20)
    data.add_argument("--data-config", default="default")

    amortizer = parser.add_argument_group("amortizer")
    amortizer.add_argument("--amortizer-name", default="PTE_CPI_24IN")
    amortizer.add_argument("--amortizer-config", default="default")
    amortizer.add_argument("--amortizer-iterations", type=int, default=100)
    amortizer.add_argument("--steps-per-iteration", type=int, default=2048)
    amortizer.add_argument("--batch-size", type=int, default=64)
    amortizer.add_argument("--save-frequency", type=int, default=10)
    return parser


def close_stage_resources() -> None:
    plt.close("all")
    gc.collect()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.skip_rl:
        if not args.rl_model:
            raise SystemExit("--rl-model is required when --skip-rl is used")
        rl_name = args.rl_model
    else:
        trainer = SACTrainer(
            num_cpu=args.rl_cpu,
            model_name_prefix=args.rl_name,
            train_config=args.rl_config,
            env_class=args.rl_env_class,
            env_setting={
                "agent_name": args.rl_agent,
                "pnc_env_name": args.rl_task,
            },
        )
        if args.rl_steps is not None:
            if args.rl_steps <= 0:
                raise SystemExit("--rl-steps must be positive")
            trainer.config.train.step = args.rl_steps
        if args.device is not None and trainer.config.model.device != args.device:
            trainer.config.model.device = args.device
            trainer.update_trainer()
        rl_name = trainer.model_name
        print(f"[1/3] RL training: {rl_name}")
        trainer.run()
        trainer.env.close()
        trainer.eval_env.close()
        close_stage_resources()

    simulator_config = Box(
        {
            "model_name": rl_name,
            "ckpt": args.rl_checkpoint,
            "traj_downsample": args.trajectory_downsample,
            "simul_option": "default",
            "fix_per_param": [],
        }
    )

    if not args.skip_data:
        print(f"[2/3] Dataset synthesis from {rl_name} ({args.rl_checkpoint})")
        validation = PnCAmortizerValidationDataset(
            load_existing_data=False,
            simulator_config=simulator_config,
            normalize_config=args.data_config,
            total_user=args.validation_users,
            trial_per_user=args.validation_trials,
        )
        validation._generate_raw_dataset(num_cpu=args.data_cpu)
        del validation

        training = PnCAmortizerTrainingDataset(
            load_existing_data=False,
            simulator_config=simulator_config,
            normalize_config=args.data_config,
            num_of_epsiode_per_param=args.trials_per_param,
        )
        training._generate_raw_dataset(
            n_param=args.train_params,
            repeat=args.data_repeat,
            num_cpu=args.data_cpu,
        )
        del training
        close_stage_resources()

    if not args.skip_amortizer:
        amortizer_name = (
            f"{args.amortizer_name}_{get_current_time_digit()}"
            if args.amortizer_name
            else None
        )
        print(f"[3/3] Amortizer training: {amortizer_name}")
        amortizer = PnCAmortizerTrainer(
            name=amortizer_name,
            train_config=args.amortizer_config,
            simul_config=simulator_config,
            data_config=args.data_config,
            train_trials_per_param=args.trials_per_param,
            validation_users=args.validation_users,
            validation_trials=args.validation_trials,
        )
        amortizer.train(
            n_iter=args.amortizer_iterations,
            step_per_iter=args.steps_per_iteration,
            batch_sz=args.batch_size,
            n_trial=args.trials_per_param,
            save_freq=args.save_frequency,
        )
        close_stage_resources()

    print(f"Pipeline complete. RL run: {rl_name}")
    print(f"Artifacts: {ROOT / 'data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
