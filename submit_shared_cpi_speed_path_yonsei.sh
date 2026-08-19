#!/usr/bin/env bash
# Curvature-free control for the radius-matched Elisa RL policy.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

MODE="${1:-smoke}"
RL_MODEL="${RL_MODEL:?Set RL_MODEL to the trained radius-matched Elisa policy}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
RUN_TAG="${RUN_TAG:-$(date +%y%m%d_%H%M%S)}"
AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_SHARED_CPI_SPEED_PATH_ELISA_RL_${RUN_TAG}}"
REUSE_DATA="${REUSE_DATA:-1}"

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
    printf 'Usage: bash %s [smoke|full]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac

export_values="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT"
export_values+=",VARIANT=summary-only,FEATURE_PROFILE=speed_path"
export_values+=",DATA_CPU=${DATA_CPU:-64},TRAIN_MAIN_PARAMS=$TRAIN_MAIN_PARAMS"
export_values+=",TRIALS_PER_BLOCK=$TRIALS_PER_BLOCK"
export_values+=",VALIDATION_USERS=$VALIDATION_USERS,VALIDATION_TRIALS=$VALIDATION_TRIALS"
export_values+=",TRAJECTORY_HZ=20,AMORTIZER_ITERATIONS=$AMORTIZER_ITERATIONS"
export_values+=",STEPS_PER_ITERATION=$STEPS_PER_ITERATION,BATCH_SIZE=$BATCH_SIZE"
export_values+=",SAVE_FREQUENCY=$SAVE_FREQUENCY,AMORTIZER_NAME=$AMORTIZER_NAME"
export_values+=",SKIP_DATA=$REUSE_DATA,SKIP_AMORTIZER=0"

submission="$(
  sbatch --parsable \
    --job-name="pnc_speed_path_${MODE}" \
    --export="$export_values" \
    run_shared_cpi_2step_yonsei.slurm
)"
job_id="${submission%%;*}"
printf '%s\n' "$job_id" > .last_shared_cpi_speed_path_job

printf 'Submitted speed/path-only shared-CPI amortizer (%s).\n' "$MODE"
printf 'RL model: %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf 'Curvature inputs: none; raw trajectory encoder: off\n'
printf 'Reuse existing raw dataset: %s\n' "$REUSE_DATA"
printf 'Slurm job: %s\n' "$job_id"
printf 'Progress: tail -F slurm-pnc-shared-cpi-%s.out\n' "$job_id"
printf 'Errors: tail -F slurm-pnc-shared-cpi-%s.err\n' "$job_id"
