"""Generate matched simulation profiles at 400, 800, and 1600 CPI.

The non-CPI parameters are shared within each simulated participant.  Only
CPI changes between its three blocks.  This makes within-profile CPI trends
directly comparable with the human experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import yaml

from ..inference.pnc_dataset_2step import (
    PnCSimulatorForConditionalAmortizer,
)
from .shared_cpi_2step_run import (
    BLOCK_PARAMETERS,
    SHARED_PARAMETERS,
    build_simulator_config,
    parse_checkpoint,
)


def progress(percent: int, message: str) -> None:
    percent = int(np.clip(percent, 0, 100))
    width = 30
    filled = round(width * percent / 100)
    bar = "#" * filled + "-" * (width - filled)
    print(f"SIMULATION  [{bar}] {percent:3d}%  {message}", flush=True)


def linear_normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        raise ValueError("Invalid CPI parameter range")
    return 2.0 * (float(value) - minimum) / (maximum - minimum) - 1.0


def build_simulator(args: argparse.Namespace) -> PnCSimulatorForConditionalAmortizer:
    config = build_simulator_config(
        args.rl_model,
        args.rl_checkpoint,
        args.trajectory_downsample,
    )
    return PnCSimulatorForConditionalAmortizer(
        model_name=config.model_name,
        ckpt=config.ckpt,
        downsample_rate=config.traj_downsample,
        simulation_option=config.simul_option,
        fix_per_param=config.fix_per_param,
        main_param_list=config.main_param,
        sub_param_list=config.sub_param,
    )


def simulate_profile(
    simulator: PnCSimulatorForConditionalAmortizer,
    cpi_values: tuple[int, ...],
    trials_per_cpi: int,
    seed: int,
) -> tuple[np.ndarray, list[pd.DataFrame]]:
    """Simulate one shared participant at fixed physical CPI values."""
    np.random.seed(seed)
    n_conditions = len(cpi_values)

    user_main_z = np.asarray(simulator.simulator.env._z_sampler(), dtype=float)
    simul_main_z = np.random.uniform(
        low=-1,
        high=1,
        size=len(simulator.simul_target_param),
    )
    user_z = np.asarray(
        [simulator.simulator.env._z_sampler() for _ in range(n_conditions)],
        dtype=float,
    )
    simul_z = np.random.uniform(
        low=-1,
        high=1,
        size=(n_conditions, len(simulator.simul_target_param)),
    )

    for parameter in simulator.main_param_list:
        if parameter in simulator.user_target_param:
            index = simulator.user_target_param.index(parameter)
            user_z[:, index] = user_main_z[index]
        elif parameter in simulator.simul_target_param:
            index = simulator.simul_target_param.index(parameter)
            simul_z[:, index] = simul_main_z[index]
        else:
            raise ValueError(f"Unknown shared parameter: {parameter}")

    if BLOCK_PARAMETERS != ("cpi",):
        raise RuntimeError("This calibration generator expects CPI as the only block parameter")
    if "cpi" not in simulator.user_target_param:
        raise RuntimeError("CPI is not present in the simulator user parameter list")
    cpi_index = simulator.user_target_param.index("cpi")
    cpi_config = simulator.simulator.env.user.param_modul["cpi"]
    cpi_min = float(cpi_config.min)
    cpi_max = float(cpi_config.max)
    for condition_index, cpi in enumerate(cpi_values):
        if not cpi_min <= cpi <= cpi_max:
            raise ValueError(f"CPI {cpi} is outside [{cpi_min}, {cpi_max}]")
        user_z[condition_index, cpi_index] = linear_normalize(
            cpi,
            cpi_min,
            cpi_max,
        )

    summaries: list[pd.DataFrame] = []
    for condition_index in range(n_conditions):
        simulator.simulator.fix_user_param(
            param_z=dict(zip(simulator.user_target_param, user_z[condition_index]))
        )
        simulator.simulator.fix_simul_param(
            param_z=dict(zip(simulator.simul_target_param, simul_z[condition_index]))
        )
        task_list = simulator.game.sample_task_conditions(
            trials_per_cpi,
            fix_cond_list=simulator.fixed_condition_per_param,
            **simulator.task_cond,
        )
        user_stat_list = simulator.simulator.sample_user_stat_conditions(
            trials_per_cpi,
            **simulator.user_stat_cond,
        )
        simulator.simulator.simulate(
            task_list=task_list,
            user_stat_list=user_stat_list,
            verbose=False,
        )
        summaries.append(simulator.simulator.get_simulation_result(return_traj=False))
        simulator.simulator.clear_simulation()

    return np.concatenate((user_z, simul_z), axis=1), summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl-model", required=True)
    parser.add_argument("--rl-checkpoint", type=parse_checkpoint, default="best")
    parser.add_argument("--profiles", type=int, default=100)
    parser.add_argument("--trials-per-cpi", type=int, default=64)
    parser.add_argument("--cpi-values", type=int, nargs="+", default=[400, 800, 1600])
    parser.add_argument("--num-cpu", type=int, default=8)
    parser.add_argument("--trajectory-downsample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.profiles <= 0 or args.trials_per_cpi <= 0 or args.num_cpu <= 0:
        raise SystemExit("--profiles, --trials-per-cpi and --num-cpu must be positive")
    cpi_values = tuple(int(value) for value in args.cpi_values)
    if len(cpi_values) != 3 or len(set(cpi_values)) != 3:
        raise SystemExit("Specify exactly three distinct --cpi-values")
    if tuple(sorted(cpi_values)) != cpi_values:
        raise SystemExit("--cpi-values must be in increasing order")

    output_path = args.output_path.resolve()
    if output_path.suffix.lower() != ".npz":
        raise SystemExit("--output-path must end in .npz")
    if output_path.exists() or output_path.with_suffix(".yaml").exists():
        raise FileExistsError(f"Calibration dataset already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    progress(0, "loading the trained RL simulator")
    simulator = build_simulator(args)
    progress(
        8,
        f"simulating {args.profiles} profiles x {len(cpi_values)} CPI blocks "
        f"x {args.trials_per_cpi} trials",
    )

    results: list[tuple[np.ndarray, list[pd.DataFrame]]] = []
    # Complete the work in chunks so the percentage reflects finished profiles,
    # rather than merely jobs submitted to the process pool.
    chunk_size = max(args.num_cpu, math.ceil(args.profiles / 20))
    for start in range(0, args.profiles, chunk_size):
        stop = min(args.profiles, start + chunk_size)
        chunk = Parallel(n_jobs=min(args.num_cpu, stop - start))(
            delayed(simulate_profile)(
                simulator,
                cpi_values,
                args.trials_per_cpi,
                args.seed + profile_index * 1009,
            )
            for profile_index in range(start, stop)
        )
        results.extend(chunk)
        percent = 8 + round(80 * stop / args.profiles)
        progress(percent, f"completed {stop}/{args.profiles} shared profiles")

    progress(90, "assembling fixed-CPI calibration dataset")
    first_summary = results[0][1][0]
    params = np.concatenate([result[0] for result in results], axis=0).astype(
        np.float32
    )
    stat_data = np.asarray(
        [
            summary.to_numpy(dtype=np.float32)
            for _, participant_summaries in results
            for summary in participant_summaries
        ],
        dtype=np.float32,
    )
    np.savez_compressed(output_path, params=params, stat_data=stat_data)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "stat": list(first_summary.columns),
        "traj": [],
        "fixed_cond": list(simulator.fixed_condition_per_param),
        "design": {
            "shared_parameters": list(SHARED_PARAMETERS),
            "block_parameter": "cpi",
            "cpi_values": list(cpi_values),
            "profiles": int(args.profiles),
            "trials_per_cpi": int(args.trials_per_cpi),
            "rl_model": args.rl_model,
            "rl_checkpoint": args.rl_checkpoint,
            "seed": int(args.seed),
        },
    }
    output_path.with_suffix(".yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        **metadata["design"],
        "created_local": datetime.now().astimezone().isoformat(),
        "npz": str(output_path),
        "condition_count": int(len(params)),
        "trial_count": int(len(params) * args.trials_per_cpi),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    progress(100, "fixed-CPI simulation dataset complete")
    print(f"FIXED_CPI_DATASET={output_path}", flush=True)
    print(f"Shapes: params={params.shape}, stat_data={stat_data.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
