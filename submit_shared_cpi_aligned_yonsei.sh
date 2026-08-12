#!/usr/bin/env bash
# Run aligned2/aligned3 with Elisa's shared-CPI amortizer settings unchanged.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

PROFILE="${1:-aligned2}"
MODE="${2:-smoke}"
RL_MODEL="${RL_MODEL:-pnc_rl_v12_cpifix_26204ASOS}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
TRAJECTORY_HZ="${TRAJECTORY_HZ:-20}"
RUN_TAG="${RUN_TAG:-$(date +%y%m%d_%H%M%S)}"
AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_SHARED_CPI_2STEP_20HZ_${PROFILE^^}_${RUN_TAG}}"
REUSE_DATA="${REUSE_DATA:-1}"

case "$PROFILE" in
  aligned2|aligned3) ;;
  *)
    printf 'Usage: bash %s [aligned2|aligned3] [smoke|full]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac

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
    ;;
  full)
    TRAIN_MAIN_PARAMS="${TRAIN_MAIN_PARAMS:-43690}"
    TRIALS_PER_BLOCK="${TRIALS_PER_BLOCK:-64}"
    VALIDATION_USERS="${VALIDATION_USERS:-100}"
    VALIDATION_TRIALS="${VALIDATION_TRIALS:-256}"
    AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-200}"
    STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-2048}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    SAVE_FREQUENCY="${SAVE_FREQUENCY:-10}"
    ;;
  *)
    printf 'Usage: bash %s [aligned2|aligned3] [smoke|full]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac

if [[ "$REUSE_DATA" != "0" && "$REUSE_DATA" != "1" ]]; then
  printf 'REUSE_DATA must be 0 or 1, got: %s\n' "$REUSE_DATA" >&2
  exit 2
fi

checkpoint_path="data/pnc_model/$RL_MODEL/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip"
if [[ ! -f "$checkpoint_path" ]]; then
  printf 'Missing RL checkpoint: %s\n' "$checkpoint_path" >&2
  exit 1
fi

export_values="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT"
export_values+=",VARIANT=summary-only,FEATURE_PROFILE=$PROFILE"
export_values+=",DATA_CPU=${DATA_CPU:-64},TRAIN_MAIN_PARAMS=$TRAIN_MAIN_PARAMS"
export_values+=",TRIALS_PER_BLOCK=$TRIALS_PER_BLOCK"
export_values+=",VALIDATION_USERS=$VALIDATION_USERS,VALIDATION_TRIALS=$VALIDATION_TRIALS"
export_values+=",TRAJECTORY_HZ=$TRAJECTORY_HZ"
export_values+=",AMORTIZER_ITERATIONS=$AMORTIZER_ITERATIONS"
export_values+=",STEPS_PER_ITERATION=$STEPS_PER_ITERATION,BATCH_SIZE=$BATCH_SIZE"
export_values+=",SAVE_FREQUENCY=$SAVE_FREQUENCY,AMORTIZER_NAME=$AMORTIZER_NAME"
export_values+=",SKIP_DATA=$REUSE_DATA,SKIP_AMORTIZER=0"

submission="$(
  sbatch --parsable \
    --job-name="pnc_${PROFILE}_${MODE}" \
    --export="$export_values" \
    run_shared_cpi_2step_yonsei.slurm
)"
job_id="${submission%%;*}"
printf '%s\n' "$job_id" > ".last_shared_cpi_${PROFILE}_job"

printf 'Submitted %s shared-CPI amortizer (%s).\n' "$PROFILE" "$MODE"
printf 'RL model: %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf 'Elisa amortizer settings retained; only feature profile changes.\n'
printf 'Reuse existing raw dataset: %s\n' "$REUSE_DATA"
printf 'Slurm job: %s\n' "$job_id"
printf 'Progress: tail -F slurm-pnc-shared-cpi-%s.out\n' "$job_id"
