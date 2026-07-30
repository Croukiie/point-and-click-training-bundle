#!/usr/bin/env bash
#SBATCH --job-name=pnc_amortizer
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-pnc-amortizer-%j.out
#SBATCH --error=slurm-pnc-amortizer-%j.err

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

: "${RL_MODEL:?RL_MODEL must name the completed RL run}"

RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
DATA_CPU="${DATA_CPU:-${SLURM_CPUS_PER_TASK:-64}}"
TRAIN_PARAMS="${TRAIN_PARAMS:-131072}"
TRIALS_PER_PARAM="${TRIALS_PER_PARAM:-64}"
VALIDATION_USERS="${VALIDATION_USERS:-100}"
VALIDATION_TRIALS="${VALIDATION_TRIALS:-256}"
TRAJECTORY_DOWNSAMPLE="${TRAJECTORY_DOWNSAMPLE:-20}"
DATA_CONFIG="${DATA_CONFIG:-curvature}"
AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_CPI_CURVATURE_24FHD}"
AMORTIZER_CONFIG="${AMORTIZER_CONFIG:-default}"
AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-200}"
STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-2048}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-10}"

"$PYTHON_BIN" - <<'PY'
import einops
import joblib
import numpy
import pandas
import seaborn
import torch

print(f"NumPy: {numpy.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this Slurm allocation.")
PY

if [[ ! -f "data/pnc_model/$RL_MODEL/best/best_model.zip" ]]; then
  printf 'Missing best RL checkpoint: %s\n' \
    "data/pnc_model/$RL_MODEL/best/best_model.zip" >&2
  exit 1
fi

command=(
  "$PYTHON_BIN" -m src.scripts.all_in_one_run
  --skip-rl
  --rl-model "$RL_MODEL"
  --rl-checkpoint "$RL_CHECKPOINT"
  --data-cpu "$DATA_CPU"
  --train-params "$TRAIN_PARAMS"
  --trials-per-param "$TRIALS_PER_PARAM"
  --validation-users "$VALIDATION_USERS"
  --validation-trials "$VALIDATION_TRIALS"
  --trajectory-downsample "$TRAJECTORY_DOWNSAMPLE"
  --data-config "$DATA_CONFIG"
  --amortizer-name "$AMORTIZER_NAME"
  --amortizer-config "$AMORTIZER_CONFIG"
  --amortizer-iterations "$AMORTIZER_ITERATIONS"
  --steps-per-iteration "$STEPS_PER_ITERATION"
  --batch-size "$BATCH_SIZE"
  --save-frequency "$SAVE_FREQUENCY"
)

if [[ "$DATA_CONFIG" == "curvature" ]]; then
  command+=(--analyze-curvature)
fi

printf 'Launching:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
