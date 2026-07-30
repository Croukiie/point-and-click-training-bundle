#!/usr/bin/env bash
# Submit RL followed by dataset synthesis and amortizer training.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

if [[ ! -f "run_bundle_yonsei.slurm" ]] \
  || [[ ! -f "run_bundle_amortizer_yonsei.slurm" ]]; then
  printf 'Run this command from the bundle directory.\n' >&2
  exit 1
fi

PIPELINE_TAG="${PIPELINE_TAG:-$(date +%y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-pnc_bundle_${PIPELINE_TAG}}"
AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_CPI_24IN_${PIPELINE_TAG}}"

STEPS="${STEPS:-30000000}"
NUM_CPU="${NUM_CPU:-64}"
RL_SETTING="${RL_SETTING:-default}"
PNC_TYPE="${PNC_TYPE:-default}"
AGENT_TYPE="${AGENT_TYPE:-classic}"

TRAIN_PARAMS="${TRAIN_PARAMS:-131072}"
TRIALS_PER_PARAM="${TRIALS_PER_PARAM:-64}"
VALIDATION_USERS="${VALIDATION_USERS:-100}"
VALIDATION_TRIALS="${VALIDATION_TRIALS:-256}"
AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-200}"

rl_export="ALL,RUN_NAME=$RUN_NAME,NO_TIME_TAG=1,STEPS=$STEPS,NUM_CPU=$NUM_CPU,RL_SETTING=$RL_SETTING,PNC_TYPE=$PNC_TYPE,AGENT_TYPE=$AGENT_TYPE"
rl_submission="$(
  sbatch --parsable \
    --export="$rl_export" \
    run_bundle_yonsei.slurm
)"
rl_job_id="${rl_submission%%;*}"

amortizer_export="ALL,RL_MODEL=$RUN_NAME,AMORTIZER_NAME=$AMORTIZER_NAME,TRAIN_PARAMS=$TRAIN_PARAMS,TRIALS_PER_PARAM=$TRIALS_PER_PARAM,VALIDATION_USERS=$VALIDATION_USERS,VALIDATION_TRIALS=$VALIDATION_TRIALS,AMORTIZER_ITERATIONS=$AMORTIZER_ITERATIONS"
amortizer_submission="$(
  sbatch --parsable \
    --dependency="afterok:$rl_job_id" \
    --export="$amortizer_export" \
    run_bundle_amortizer_yonsei.slurm
)"
amortizer_job_id="${amortizer_submission%%;*}"

printf 'Pipeline submitted successfully.\n'
printf 'RL run: %s\n' "$RUN_NAME"
printf 'RL job: %s\n' "$rl_job_id"
printf 'Dataset + amortizer job: %s (starts after RL succeeds)\n' \
  "$amortizer_job_id"
printf 'Amortizer prefix: %s\n' "$AMORTIZER_NAME"
