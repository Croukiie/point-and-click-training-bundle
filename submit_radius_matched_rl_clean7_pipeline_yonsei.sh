#!/usr/bin/env bash
# Train Elisa's radius-matched CPI RL policy, then train the shared clean7
# amortizer only if RL training succeeds.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

MODE="${1:-smoke}"
RUN_TAG="${RUN_TAG:-$(date +%y%m%d_%H%M%S)}"
RL_MODEL="${RL_MODEL:-pnc_rl_elisa_radius_matched_${RUN_TAG}}"
AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_SHARED_CPI_CLEAN7_ELISA_RL_${RUN_TAG}}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

case "$MODE" in
  smoke)
    RL_STEPS="${RL_STEPS:-100000}"
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
    RL_STEPS="${RL_STEPS:-30000000}"
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

rl_export="ALL,PYTHON_BIN=$PYTHON_BIN,RUN_NAME=$RL_MODEL,NO_TIME_TAG=1"
rl_export+=",STEPS=$RL_STEPS,NUM_CPU=${NUM_CPU:-64},RL_SETTING=default"
rl_export+=",PNC_TYPE=radius_matched_cpi,AGENT_TYPE=classic,DEVICE=cuda"
rl_submission="$(sbatch --parsable --export="$rl_export" run_bundle_yonsei.slurm)"
rl_job_id="${rl_submission%%;*}"

amortizer_export="ALL,PYTHON_BIN=$PYTHON_BIN,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=best"
amortizer_export+=",VARIANT=summary-only,FEATURE_PROFILE=clean7,SKIP_DATA=0,SKIP_AMORTIZER=0"
amortizer_export+=",DATA_CPU=${DATA_CPU:-64},TRAIN_MAIN_PARAMS=$TRAIN_MAIN_PARAMS"
amortizer_export+=",TRIALS_PER_BLOCK=$TRIALS_PER_BLOCK,VALIDATION_USERS=$VALIDATION_USERS"
amortizer_export+=",VALIDATION_TRIALS=$VALIDATION_TRIALS,TRAJECTORY_HZ=20"
amortizer_export+=",AMORTIZER_ITERATIONS=$AMORTIZER_ITERATIONS"
amortizer_export+=",STEPS_PER_ITERATION=$STEPS_PER_ITERATION,BATCH_SIZE=$BATCH_SIZE"
amortizer_export+=",SAVE_FREQUENCY=$SAVE_FREQUENCY,AMORTIZER_NAME=$AMORTIZER_NAME"
amortizer_submission="$(
  sbatch --parsable \
    --dependency="afterok:$rl_job_id" \
    --job-name="elisa_clean7_${MODE}" \
    --export="$amortizer_export" \
    run_shared_cpi_2step_yonsei.slurm
)"
amortizer_job_id="${amortizer_submission%%;*}"

printf '%s %s\n' "$rl_job_id" "$amortizer_job_id" > .last_radius_matched_pipeline_jobs

printf 'Submitted Elisa radius-matched CPI pipeline (%s).\n' "$MODE"
printf 'Target radius: 0.00125-0.007 m\n'
printf 'RL model: %s\n' "$RL_MODEL"
printf 'RL job: %s\n' "$rl_job_id"
printf 'Shared clean7 amortizer: %s\n' "$AMORTIZER_NAME"
printf 'Amortizer job: %s (afterok:%s)\n' "$amortizer_job_id" "$rl_job_id"
printf 'Status: squeue -j %s,%s\n' "$rl_job_id" "$amortizer_job_id"
