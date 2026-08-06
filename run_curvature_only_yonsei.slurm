#!/usr/bin/env bash
#SBATCH --job-name=pnc_curv_only
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-pnc-curvature-only-%j.out
#SBATCH --error=slurm-pnc-curvature-only-%j.err

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BUNDLE_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3.11}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_26210DBCT}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
AMORTIZER_PREFIX="${AMORTIZER_PREFIX:-PTE_CPI_CURVATURE_ONLY_24FHD}"
AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-200}"
STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-2048}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRIALS_PER_PARAM="${TRIALS_PER_PARAM:-64}"
VALIDATION_USERS="${VALIDATION_USERS:-100}"
VALIDATION_TRIALS="${VALIDATION_TRIALS:-256}"

if [[ ! -f "data/pnc_model/$RL_MODEL/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip" ]]; then
  printf 'Missing RL checkpoint for %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT" >&2
  exit 1
fi

"$PYTHON_BIN" -u - <<PY
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import torch
from box import Box

from src.config.config import INF
from src.inference.pnc_infer_trainer import PnCAmortizerTrainer
from src.utils.myutils import get_current_time_digit

rl_model = ${RL_MODEL@Q}
rl_checkpoint = ${RL_CHECKPOINT@Q}
name = f"${AMORTIZER_PREFIX}_{get_current_time_digit()}"

simulator_config = Box({
    "model_name": rl_model,
    "ckpt": rl_checkpoint,
    "traj_downsample": 20,
    "simul_option": "default",
    "fix_per_param": [],
})

# Reuse every curvature statistic already stored in the completed dataset,
# while disabling the expensive raw 20 Hz trajectory encoder.
data_config = deepcopy(INF.normalize.curvature)
data_config.traj.ignore_traj = True

print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this Slurm allocation.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Curvature-only run: {name}", flush=True)
print("Reusing the existing 20 Hz dataset; raw trajectory encoder disabled.", flush=True)

trainer = PnCAmortizerTrainer(
    name=name,
    train_config="default",
    simul_config=simulator_config,
    data_config=data_config,
    train_trials_per_param=int("${TRIALS_PER_PARAM}"),
    validation_users=int("${VALIDATION_USERS}"),
    validation_trials=int("${VALIDATION_TRIALS}"),
)
trainer.train(
    n_iter=int("${AMORTIZER_ITERATIONS}"),
    step_per_iter=int("${STEPS_PER_ITERATION}"),
    batch_sz=int("${BATCH_SIZE}"),
    n_trial=int("${TRIALS_PER_PARAM}"),
    save_freq=10,
)
print(f"CURVATURE_ONLY_COMPLETE={name}", flush=True)
PY
