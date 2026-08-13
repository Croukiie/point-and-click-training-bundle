#!/usr/bin/env bash
#SBATCH --job-name=pnc_hidden_core3
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-pnc-hidden-core3-%j.out
#SBATCH --error=slurm-pnc-hidden-core3-%j.err

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BUNDLE_DIR"

if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3.11}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

RL_MODEL="${RL_MODEL:?RL_MODEL is required}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
DATA_CPU="${DATA_CPU:-${SLURM_CPUS_PER_TASK:-64}}"
TRAIN_MAIN_PARAMS="${TRAIN_MAIN_PARAMS:?TRAIN_MAIN_PARAMS is required}"
TRIALS_PER_BLOCK="${TRIALS_PER_BLOCK:?TRIALS_PER_BLOCK is required}"
VALIDATION_USERS="${VALIDATION_USERS:?VALIDATION_USERS is required}"
VALIDATION_TRIALS="${VALIDATION_TRIALS:?VALIDATION_TRIALS is required}"
AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:?AMORTIZER_ITERATIONS is required}"
STEPS_PER_ITERATION="${STEPS_PER_ITERATION:?STEPS_PER_ITERATION is required}"
BATCH_SIZE="${BATCH_SIZE:?BATCH_SIZE is required}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:?SAVE_FREQUENCY is required}"
AMORTIZER_NAME="${AMORTIZER_NAME:?AMORTIZER_NAME is required}"

model_dir="data/pnc_model/$RL_MODEL"
for required in \
  "$model_dir/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip" \
  "$model_dir/env_info.yaml" \
  "$model_dir/train_config.yaml"
do
  if [[ ! -f "$required" ]]; then
    printf 'Missing RL output: %s\n' "$required" >&2
    exit 1
  fi
done

env_class="$(awk -F': *' '/^env_class:/{print $2; exit}' "$model_dir/env_info.yaml" | tr -d '\r')"
if [[ "$env_class" != PnCEnvCpiHidden ]]; then
  printf 'Expected PnCEnvCpiHidden, found: <%s>\n' "${env_class:-missing}" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

printf 'RL model: %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf '%s\n' 'RL environment: PnCEnvCpiHidden'
printf '%s\n' 'Amortizer: shared-CPI, summary-only, true CPI hidden, 20 Hz'
printf '%s\n' 'Curvature CORE3: turn energy, early turn, late turn'

"$PYTHON_BIN" -u -m src.scripts.shared_cpi_core3_run \
  --rl-model "$RL_MODEL" \
  --rl-checkpoint "$RL_CHECKPOINT" \
  --variant summary-only \
  --feature-profile core3 \
  --data-cpu "$DATA_CPU" \
  --train-main-params "$TRAIN_MAIN_PARAMS" \
  --blocks-per-participant 3 \
  --trials-per-block "$TRIALS_PER_BLOCK" \
  --validation-users "$VALIDATION_USERS" \
  --validation-trials "$VALIDATION_TRIALS" \
  --trajectory-downsample 20 \
  --amortizer-name "$AMORTIZER_NAME" \
  --amortizer-iterations "$AMORTIZER_ITERATIONS" \
  --steps-per-iteration "$STEPS_PER_ITERATION" \
  --batch-size "$BATCH_SIZE" \
  --save-frequency "$SAVE_FREQUENCY" \
  --device cuda
