#!/usr/bin/env bash
# Generate one matched 20 Hz dataset, then train trajectory and summary-only
# shared-CPI amortizers from exactly the same simulated trials.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

MODE="${1:-smoke}"
RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_hidden_short}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
TRAJECTORY_HZ="${TRAJECTORY_HZ:-20}"
PAIR_TAG="${PAIR_TAG:-$(date +%y%m%d_%H%M%S)}"
AMORTIZER_BASE="${AMORTIZER_BASE:-PTE_SHARED_CPI_2STEP_20HZ_${PAIR_TAG}}"

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

checkpoint_path="data/pnc_model/$RL_MODEL/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip"
if [[ ! -f "$checkpoint_path" ]]; then
  printf 'Missing RL checkpoint: %s\n' "$checkpoint_path" >&2
  printf 'Available best checkpoints:\n' >&2
  find data/pnc_model -path '*/best/best_model.zip' -print 2>/dev/null >&2 || true
  exit 1
fi

common="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT"
common+=",DATA_CPU=${DATA_CPU:-64},TRAIN_MAIN_PARAMS=$TRAIN_MAIN_PARAMS"
common+=",TRIALS_PER_BLOCK=$TRIALS_PER_BLOCK"
common+=",VALIDATION_USERS=$VALIDATION_USERS,VALIDATION_TRIALS=$VALIDATION_TRIALS"
common+=",TRAJECTORY_HZ=$TRAJECTORY_HZ"
common+=",AMORTIZER_ITERATIONS=$AMORTIZER_ITERATIONS"
common+=",STEPS_PER_ITERATION=$STEPS_PER_ITERATION,BATCH_SIZE=$BATCH_SIZE"
common+=",SAVE_FREQUENCY=$SAVE_FREQUENCY,AMORTIZER_NAME=$AMORTIZER_BASE"

data_submission="$(
  sbatch --parsable \
    --job-name="pnc_20hz_data_${MODE}" \
    --export="$common,VARIANT=trajectory,SKIP_DATA=0,SKIP_AMORTIZER=1" \
    run_shared_cpi_2step_yonsei.slurm
)"
data_job="${data_submission%%;*}"

trajectory_submission="$(
  sbatch --parsable \
    --dependency="afterok:$data_job" \
    --job-name="pnc_20hz_traj_${MODE}" \
    --export="$common,VARIANT=trajectory,SKIP_DATA=1,SKIP_AMORTIZER=0" \
    run_shared_cpi_2step_yonsei.slurm
)"
trajectory_job="${trajectory_submission%%;*}"

summary_submission="$(
  sbatch --parsable \
    --dependency="afterok:$data_job" \
    --job-name="pnc_20hz_summary_${MODE}" \
    --export="$common,VARIANT=summary-only,SKIP_DATA=1,SKIP_AMORTIZER=0" \
    run_shared_cpi_2step_yonsei.slurm
)"
summary_job="${summary_submission%%;*}"

printf '%s %s %s\n' "$data_job" "$trajectory_job" "$summary_job" \
  > .last_shared_cpi_pair_jobs

printf 'Submitted matched 20 Hz shared-CPI pair (%s).\n' "$MODE"
printf 'RL model: %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf 'Dataset job:      %s\n' "$data_job"
printf 'Trajectory job:   %s (after dataset)\n' "$trajectory_job"
printf 'Summary-only job: %s (after dataset)\n' "$summary_job"
printf 'Amortizer base:   %s\n' "$AMORTIZER_BASE"
printf 'Status: squeue -j %s,%s,%s\n' "$data_job" "$trajectory_job" "$summary_job"
printf 'Dataset progress: tail -F slurm-pnc-shared-cpi-%s.out\n' "$data_job"
printf 'Trajectory log:   tail -F slurm-pnc-shared-cpi-%s.out\n' "$trajectory_job"
printf 'Summary log:      tail -F slurm-pnc-shared-cpi-%s.out\n' "$summary_job"
