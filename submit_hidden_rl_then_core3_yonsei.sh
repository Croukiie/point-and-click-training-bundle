#!/usr/bin/env bash
# Submit: 30M hidden-CPI RL -> CORE3 smoke -> CORE3 full amortizer.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

for required in \
  run_bundle_yonsei.slurm \
  run_hidden_cpi_core3_yonsei.slurm \
  src/scripts/shared_cpi_core3_run.py
do
  if [[ ! -f "$required" ]]; then
    printf 'Missing required file: %s\n' "$required" >&2
    exit 1
  fi
done

RUN_TAG="${RUN_TAG:-$(date +%y%m%d_%H%M%S)}"
RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_hidden_yonsei_30m_${RUN_TAG}}"
RL_STEPS="${RL_STEPS:-30000000}"
RL_CPUS="${RL_CPUS:-64}"
RL_MEM="${RL_MEM:-32G}"

if [[ -e "data/pnc_model/$RL_MODEL" ]]; then
  printf 'RL output directory already exists: data/pnc_model/%s\n' "$RL_MODEL" >&2
  printf '%s\n' 'Choose another RUN_TAG or RL_MODEL.' >&2
  exit 1
fi

printf '[1/3] Submitting hidden-CPI RL training (%s steps)...\n' "$RL_STEPS"
rl_submission="$(sbatch --parsable \
  --job-name=pnc_hidden_rl_30m \
  --cpus-per-task="$RL_CPUS" \
  --mem="$RL_MEM" \
  --export="ALL,NUM_CPU=$RL_CPUS,RUN_NAME=$RL_MODEL,ENV_CLASS_TYPE=PnCEnvCpiHidden,AGENT_TYPE=classic,PNC_TYPE=default,RL_SETTING=default,STEPS=$RL_STEPS,DEVICE=cuda,NO_TIME_TAG=1" \
  run_bundle_yonsei.slurm)"
rl_job="${rl_submission%%;*}"

common="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=best"

printf '[2/3] Queueing CORE3 smoke after RL job %s...\n' "$rl_job"
smoke_submission="$(sbatch --parsable \
  --dependency="afterok:$rl_job" \
  --job-name=pnc_hid_c3_smoke \
  --cpus-per-task=8 \
  --mem=32G \
  --export="$common,DATA_CPU=8,TRAIN_MAIN_PARAMS=12,TRIALS_PER_BLOCK=8,VALIDATION_USERS=8,VALIDATION_TRIALS=16,AMORTIZER_ITERATIONS=2,STEPS_PER_ITERATION=32,BATCH_SIZE=4,SAVE_FREQUENCY=1,AMORTIZER_NAME=PTE_SHARED_CPI_HIDDEN_CORE3_20HZ_SMOKE_$RUN_TAG" \
  run_hidden_cpi_core3_yonsei.slurm)"
smoke_job="${smoke_submission%%;*}"

printf '[3/3] Queueing full CORE3 amortizer after smoke job %s...\n' "$smoke_job"
full_submission="$(sbatch --parsable \
  --dependency="afterok:$smoke_job" \
  --job-name=pnc_hid_c3_full \
  --export="$common,DATA_CPU=64,TRAIN_MAIN_PARAMS=43690,TRIALS_PER_BLOCK=64,VALIDATION_USERS=100,VALIDATION_TRIALS=256,AMORTIZER_ITERATIONS=200,STEPS_PER_ITERATION=2048,BATCH_SIZE=16,SAVE_FREQUENCY=10,AMORTIZER_NAME=PTE_SHARED_CPI_HIDDEN_CORE3_20HZ_FULL_$RUN_TAG" \
  run_hidden_cpi_core3_yonsei.slurm)"
full_job="${full_submission%%;*}"

printf '%s %s %s %s\n' "$RL_MODEL" "$rl_job" "$smoke_job" "$full_job" \
  > .last_hidden_rl_core3_chain

printf '\n%s\n' 'CHAIN_SUBMITTED'
printf 'RL model:       %s\n' "$RL_MODEL"
printf 'RL job:         %s\n' "$rl_job"
printf 'CORE3 smoke:    %s (afterok:%s)\n' "$smoke_job" "$rl_job"
printf 'CORE3 full:     %s (afterok:%s)\n' "$full_job" "$smoke_job"
printf 'Status: squeue -j %s,%s,%s -o "%%.18i %%.10T %%.12M %%.40R"\n' \
  "$rl_job" "$smoke_job" "$full_job"
printf 'RL log:    tail -F slurm-pnc-bundle-%s.out\n' "$rl_job"
printf 'Smoke log: tail -F slurm-pnc-hidden-core3-%s.out\n' "$smoke_job"
printf 'Full log:  tail -F slurm-pnc-hidden-core3-%s.out\n' "$full_job"
