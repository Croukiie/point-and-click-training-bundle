#!/usr/bin/env bash
# Submit the shared-parameter, block-specific-CPI two-step amortizer.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

MODE="${1:-smoke}"
VARIANT="${2:-trajectory}"
RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_26210DBCT}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"

case "$VARIANT" in
  trajectory|summary-only) ;;
  *)
    printf 'Variant must be trajectory or summary-only.\n' >&2
    exit 2
    ;;
esac

if [[ ! -f "data/pnc_model/$RL_MODEL/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip" ]]; then
  printf 'Missing RL checkpoint for %s (%s).\n' "$RL_MODEL" "$RL_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "run_shared_cpi_2step_yonsei.slurm" ]]; then
  printf 'Missing run_shared_cpi_2step_yonsei.slurm.\n' >&2
  exit 1
fi

case "$MODE" in
  smoke)
    TRAIN_MAIN_PARAMS="${TRAIN_MAIN_PARAMS:-12}"
    TRIALS_PER_BLOCK="${TRIALS_PER_BLOCK:-8}"
    VALIDATION_USERS="${VALIDATION_USERS:-8}"
    VALIDATION_TRIALS="${VALIDATION_TRIALS:-16}"
    AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-2}"
    STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-32}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    SAVE_FREQUENCY="${SAVE_FREQUENCY:-1}"
    AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_SHARED_CPI_2STEP_SMOKE}"
    ;;
  full)
    # 43,690 x 3 x 64 = 8,388,480 simulations, approximately the
    # same total training-trial count as the independent amortizer.
    TRAIN_MAIN_PARAMS="${TRAIN_MAIN_PARAMS:-43690}"
    TRIALS_PER_BLOCK="${TRIALS_PER_BLOCK:-64}"
    VALIDATION_USERS="${VALIDATION_USERS:-100}"
    VALIDATION_TRIALS="${VALIDATION_TRIALS:-256}"
    AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-200}"
    STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-2048}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    SAVE_FREQUENCY="${SAVE_FREQUENCY:-10}"
    AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_SHARED_CPI_2STEP_24FHD}"
    ;;
  *)
    printf 'Usage: bash %s [smoke|full] [trajectory|summary-only]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac

export_values="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT,VARIANT=$VARIANT"
export_values+=",DATA_CPU=${DATA_CPU:-64},TRAIN_MAIN_PARAMS=$TRAIN_MAIN_PARAMS"
export_values+=",TRIALS_PER_BLOCK=$TRIALS_PER_BLOCK"
export_values+=",VALIDATION_USERS=$VALIDATION_USERS,VALIDATION_TRIALS=$VALIDATION_TRIALS"
export_values+=",AMORTIZER_ITERATIONS=$AMORTIZER_ITERATIONS"
export_values+=",STEPS_PER_ITERATION=$STEPS_PER_ITERATION,BATCH_SIZE=$BATCH_SIZE"
export_values+=",SAVE_FREQUENCY=$SAVE_FREQUENCY,AMORTIZER_NAME=$AMORTIZER_NAME"
export_values+=",SKIP_DATA=${SKIP_DATA:-0}"

submission="$(
  sbatch --parsable \
    --job-name="pnc_shared_${MODE}" \
    --export="$export_values" \
    run_shared_cpi_2step_yonsei.slurm
)"
job_id="${submission%%;*}"

printf 'Submitted %s shared-CPI 2-step run (%s).\n' "$MODE" "$VARIANT"
printf 'RL model: %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf 'Slurm job: %s\n' "$job_id"
printf 'Progress with percentage:\n'
printf '  tail -F slurm-pnc-shared-cpi-%s.out\n' "$job_id"
printf 'Errors:\n'
printf '  tail -F slurm-pnc-shared-cpi-%s.err\n' "$job_id"
