#!/usr/bin/env bash
#SBATCH --job-name=pnc_raw_pair
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/pnc-rawcount-pair-%A_%a.out
#SBATCH --error=logs/pnc-rawcount-pair-%A_%a.err

set -euo pipefail
cd "${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RL_MODEL="${RL_MODEL:?RL_MODEL is required}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
RUN_STAGE="${RUN_STAGE:?RUN_STAGE is required}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required}"
PIPELINE_SEED="${PIPELINE_SEED:-20260821}"
CONTRACT="movement_exact20_endpoint_clickonly_rawcount125_v4"
PROFILES=(speedpath rawcount)
PROFILE="${PROFILES[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}"

case "$RUN_STAGE" in
  smoke) ITERATIONS=2; STEPS_PER_ITERATION=32; BATCH_SIZE=4; SAVE_FREQUENCY=1 ;;
  full) ITERATIONS=200; STEPS_PER_ITERATION=2048; BATCH_SIZE=16; SAVE_FREQUENCY=10 ;;
  *) printf 'Unknown RUN_STAGE: %s\n' "$RUN_STAGE" >&2; exit 2 ;;
esac

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

dataset_dir="data/amortizer_cond/dataset/${RL_MODEL}_${RL_CHECKPOINT}_20_default_012345678_9_${CONTRACT}"
compgen -G "$dataset_dir/tr_*mp_3sp_64ep_*.npz" >/dev/null || {
  printf 'Missing training dataset: %s\n' "$dataset_dir" >&2; exit 1;
}
compgen -G "$dataset_dir/vd_100users_3_conds_256trials_*.npz" >/dev/null || {
  printf 'Missing validation dataset: %s\n' "$dataset_dir" >&2; exit 1;
}

AMORTIZER_NAME="PTE_SHARED_CPI_RAWCOUNT_${PROFILE^^}_${RUN_STAGE^^}_${RUN_TAG}"
"$PYTHON_BIN" -u -m src.scripts.shared_cpi_rawcount_ablation \
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
  --seed "$PIPELINE_SEED"
