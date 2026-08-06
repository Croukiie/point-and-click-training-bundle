#!/usr/bin/env bash
#SBATCH --job-name=pnc_shared_cpi
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-pnc-shared-cpi-%j.out
#SBATCH --error=slurm-pnc-shared-cpi-%j.err

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
VARIANT="${VARIANT:-trajectory}"
DATA_CPU="${DATA_CPU:-${SLURM_CPUS_PER_TASK:-64}}"
TRAIN_MAIN_PARAMS="${TRAIN_MAIN_PARAMS:-43690}"
TRIALS_PER_BLOCK="${TRIALS_PER_BLOCK:-64}"
VALIDATION_USERS="${VALIDATION_USERS:-100}"
VALIDATION_TRIALS="${VALIDATION_TRIALS:-256}"
AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-200}"
STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-10}"
AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_SHARED_CPI_2STEP_24FHD}"
SKIP_DATA="${SKIP_DATA:-0}"

checkpoint_path="data/pnc_model/$RL_MODEL/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip"
if [[ ! -f "$checkpoint_path" ]]; then
  printf 'Missing RL checkpoint: %s\n' "$checkpoint_path" >&2
  exit 1
fi

printf 'Python: '
"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

args=(
  -m src.scripts.shared_cpi_2step_run
  --rl-model "$RL_MODEL"
  --rl-checkpoint "$RL_CHECKPOINT"
  --variant "$VARIANT"
  --data-cpu "$DATA_CPU"
  --train-main-params "$TRAIN_MAIN_PARAMS"
  --blocks-per-participant 3
  --trials-per-block "$TRIALS_PER_BLOCK"
  --validation-users "$VALIDATION_USERS"
  --validation-trials "$VALIDATION_TRIALS"
  --amortizer-name "$AMORTIZER_NAME"
  --amortizer-iterations "$AMORTIZER_ITERATIONS"
  --steps-per-iteration "$STEPS_PER_ITERATION"
  --batch-size "$BATCH_SIZE"
  --save-frequency "$SAVE_FREQUENCY"
  --device cuda
)

if [[ "$SKIP_DATA" == "1" ]]; then
  args+=(--skip-data)
fi

"$PYTHON_BIN" -u "${args[@]}"
