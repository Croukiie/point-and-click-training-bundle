#!/usr/bin/env bash
# Submit the CPI-curvature dataset and amortizer from the completed RL model.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

MODE="${1:-smoke}"
RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_26210DBCT}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"

if [[ ! -f "data/pnc_model/$RL_MODEL/best/best_model.zip" ]]; then
  printf 'Missing completed RL checkpoint: %s\n' \
    "data/pnc_model/$RL_MODEL/best/best_model.zip" >&2
  exit 1
fi
if [[ ! -f "run_bundle_amortizer_yonsei.slurm" ]]; then
  printf 'Missing run_bundle_amortizer_yonsei.slurm\n' >&2
  exit 1
fi

case "$MODE" in
  smoke)
    TRAIN_PARAMS="${TRAIN_PARAMS:-32}"
    TRIALS_PER_PARAM="${TRIALS_PER_PARAM:-8}"
    VALIDATION_USERS="${VALIDATION_USERS:-8}"
    VALIDATION_TRIALS="${VALIDATION_TRIALS:-16}"
    AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-2}"
    STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-32}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    SAVE_FREQUENCY="${SAVE_FREQUENCY:-1}"
    AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_CPI_CURVATURE_SMOKE}"
    ;;
  full)
    TRAIN_PARAMS="${TRAIN_PARAMS:-131072}"
    TRIALS_PER_PARAM="${TRIALS_PER_PARAM:-64}"
    VALIDATION_USERS="${VALIDATION_USERS:-100}"
    VALIDATION_TRIALS="${VALIDATION_TRIALS:-256}"
    AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-200}"
    STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-2048}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    SAVE_FREQUENCY="${SAVE_FREQUENCY:-10}"
    AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_CPI_CURVATURE_24FHD}"
    ;;
  *)
    printf 'Usage: bash %s [smoke|full]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac

export_values="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT"
export_values+=",DATA_CONFIG=curvature,TRAJECTORY_DOWNSAMPLE=20"
export_values+=",TRAIN_PARAMS=$TRAIN_PARAMS,TRIALS_PER_PARAM=$TRIALS_PER_PARAM"
export_values+=",VALIDATION_USERS=$VALIDATION_USERS,VALIDATION_TRIALS=$VALIDATION_TRIALS"
export_values+=",AMORTIZER_ITERATIONS=$AMORTIZER_ITERATIONS"
export_values+=",STEPS_PER_ITERATION=$STEPS_PER_ITERATION,BATCH_SIZE=$BATCH_SIZE"
export_values+=",SAVE_FREQUENCY=$SAVE_FREQUENCY"
export_values+=",AMORTIZER_NAME=$AMORTIZER_NAME"

submission="$(
  sbatch --parsable \
    --job-name="pnc_curv_${MODE}" \
    --export="$export_values" \
    run_bundle_amortizer_yonsei.slurm
)"
job_id="${submission%%;*}"

printf 'Submitted %s curvature run.\n' "$MODE"
printf 'RL model: %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf 'Slurm job: %s\n' "$job_id"
printf 'Progress: tail -f slurm-pnc-amortizer-%s.out\n' "$job_id"
printf 'Errors:   tail -f slurm-pnc-amortizer-%s.err\n' "$job_id"
