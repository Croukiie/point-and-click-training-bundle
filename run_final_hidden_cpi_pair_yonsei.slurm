#!/usr/bin/env bash
#SBATCH --job-name=pnc_final_pair
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/pnc-final-pair-%A_%a.out
#SBATCH --error=logs/pnc-final-pair-%A_%a.err

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BUNDLE_DIR"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RL_MODEL="${RL_MODEL:?RL_MODEL is required}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
RUN_STAGE="${RUN_STAGE:?RUN_STAGE is required}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
PAIR_SEED="${PAIR_SEED:-20260813}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

PROFILES=(speedpath core3)
PROFILE="${PROFILES[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}"

case "$RUN_STAGE" in
  smoke)
    ITERATIONS=2
    STEPS_PER_ITERATION=32
    BATCH_SIZE=4
    SAVE_FREQUENCY=1
    ;;
  full)
    ITERATIONS=200
    STEPS_PER_ITERATION=2048
    BATCH_SIZE=16
    SAVE_FREQUENCY=10
    ;;
  *)
    printf 'Unknown RUN_STAGE: %s\n' "$RUN_STAGE" >&2
    exit 2
    ;;
esac

model_dir="data/pnc_model/$RL_MODEL"
env_class="$(awk -F': *' '/^env_class:/{print $2; exit}' "$model_dir/env_info.yaml" 2>/dev/null | tr -d '\r')"
[[ "$env_class" == PnCEnvCpiHidden ]] || {
  printf 'Expected PnCEnvCpiHidden, found: <%s>\n' "${env_class:-missing}" >&2
  exit 1
}

dataset_dir="data/amortizer_cond/dataset/${RL_MODEL}_${RL_CHECKPOINT}_20_default_012345678_9"
compgen -G "$dataset_dir/tr_*mp_3sp_64ep_*.npz" >/dev/null || {
  printf 'Missing shared training data: %s\n' "$dataset_dir" >&2
  exit 1
}
compgen -G "$dataset_dir/vd_100users_3_conds_256trials_*.npz" >/dev/null || {
  printf 'Missing shared validation data: %s\n' "$dataset_dir" >&2
  exit 1
}

"$PYTHON_BIN" - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this allocation")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

AMORTIZER_NAME="PTE_SHARED_CPI_FINAL_${PROFILE^^}_${RUN_STAGE^^}_${RUN_TAG}"

printf 'FINAL_PAIR_PROFILE=%s\n' "$PROFILE"
printf 'FINAL_PAIR_STAGE=%s\n' "$RUN_STAGE"
printf 'FINAL_PAIR_SEED=%s\n' "$PAIR_SEED"
printf 'FINAL_PAIR_RL=%s\n' "$RL_MODEL"
printf 'FINAL_PAIR_DATASET=%s\n' "$dataset_dir"
printf '%s\n' 'MATCHED_FIELDS=RL,dataset,20Hz,summary-only,9+1 parameters,seed,budget'

"$PYTHON_BIN" -u -m src.scripts.shared_cpi_final_pair \
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
  --pair-seed "$PAIR_SEED"
