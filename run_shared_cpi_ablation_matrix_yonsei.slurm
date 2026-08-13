#!/usr/bin/env bash
#SBATCH --job-name=pnc_cpi_ablation
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/pnc-cpi-ablation-%A_%a.out
#SBATCH --error=logs/pnc-cpi-ablation-%A_%a.err

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

PROFILES=(speedpath core3 clean7 full10)
PROFILE="${PROFILES[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}"

RL_MODEL="${RL_MODEL:?RL_MODEL is required}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
RUN_MODE="${RUN_MODE:-smoke}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
ABLATION_SEED="${ABLATION_SEED:-20260813}"

case "$RUN_MODE" in
  smoke)
    ITERATIONS=2
    STEPS_PER_ITERATION=32
    BATCH_SIZE=4
    SAVE_FREQUENCY=1
    ;;
  diagnostic)
    ITERATIONS=100
    STEPS_PER_ITERATION=2048
    BATCH_SIZE=16
    SAVE_FREQUENCY=10
    ;;
  full)
    ITERATIONS=200
    STEPS_PER_ITERATION=2048
    BATCH_SIZE=16
    SAVE_FREQUENCY=10
    ;;
  *)
    printf 'Unknown RUN_MODE: %s\n' "$RUN_MODE" >&2
    exit 2
    ;;
esac

model_dir="data/pnc_model/$RL_MODEL"
env_class="$(awk -F': *' '/^env_class:/{print $2; exit}' "$model_dir/env_info.yaml" 2>/dev/null | tr -d '\r')"
if [[ "$env_class" != PnCEnvCpiHidden ]]; then
  printf 'Expected a hidden-CPI RL, found env_class=<%s>\n' "${env_class:-missing}" >&2
  exit 1
fi

dataset_dir="data/amortizer_cond/dataset/${RL_MODEL}_${RL_CHECKPOINT}_20_default_012345678_9"
if ! compgen -G "$dataset_dir/tr_*mp_3sp_64ep_*.npz" >/dev/null; then
  printf 'Missing shared 64-trial training data: %s\n' "$dataset_dir" >&2
  exit 1
fi
if ! compgen -G "$dataset_dir/vd_100users_3_conds_256trials_*.npz" >/dev/null; then
  printf 'Missing shared validation data: %s\n' "$dataset_dir" >&2
  exit 1
fi

AMORTIZER_NAME="PTE_SHARED_CPI_MATRIX_${PROFILE}_${RUN_MODE}_${RUN_TAG}"

printf 'ABLATION_PROFILE=%s\n' "$PROFILE"
printf 'ABLATION_MODE=%s\n' "$RUN_MODE"
printf 'ABLATION_SEED=%s\n' "$ABLATION_SEED"
printf 'RL_MODEL=%s\n' "$RL_MODEL"
printf 'DATASET=%s\n' "$dataset_dir"
printf '%s\n' 'CONTROLLED_FIELDS=RL,dataset,20Hz,summary-only,9+1 parameters,training budget,seed'

"$PYTHON_BIN" -u -m src.scripts.shared_cpi_ablation_matrix \
  --rl-model "$RL_MODEL" \
  --rl-checkpoint "$RL_CHECKPOINT" \
  --variant summary-only \
  --feature-profile "$PROFILE" \
  --skip-data \
  --train-main-params 43690 \
  --blocks-per-participant 3 \
  --trials-per-block 64 \
  --validation-users 100 \
  --validation-trials 256 \
  --trajectory-downsample 20 \
  --amortizer-name "$AMORTIZER_NAME" \
  --amortizer-iterations "$ITERATIONS" \
  --steps-per-iteration "$STEPS_PER_ITERATION" \
  --batch-size "$BATCH_SIZE" \
  --save-frequency "$SAVE_FREQUENCY" \
  --device cuda \
  --ablation-seed "$ABLATION_SEED"
