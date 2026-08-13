#!/usr/bin/env bash
# Submit a smoke test followed automatically by the full hidden-CPI CORE3 run.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_hidden_local_30m_26222DSFT}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
TRAJECTORY_HZ="${TRAJECTORY_HZ:-20}"
RUN_TAG="${RUN_TAG:-$(date +%y%m%d_%H%M%S)}"

model_dir="data/pnc_model/$RL_MODEL"
for required in \
  "$model_dir/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip" \
  "$model_dir/env_info.yaml" \
  "$model_dir/train_config.yaml"
do
  if [[ ! -f "$required" ]]; then
    printf 'Missing RL model file: %s\n' "$required" >&2
    exit 1
  fi
done

env_class="$(awk -F': *' '/^env_class:/{print $2; exit}' "$model_dir/env_info.yaml" | tr -d '\r')"
if [[ "$env_class" != "PnCEnvCpiHidden" ]]; then
  printf 'Expected PnCEnvCpiHidden, found: %s\n' "${env_class:-unknown}" >&2
  exit 1
fi

if [[ ! -f src/scripts/shared_cpi_core3_run.py ]] || \
   [[ ! -f run_hidden_cpi_core3_yonsei.slurm ]]; then
  printf '%s\n' 'The CORE3 update is incomplete.' >&2
  exit 1
fi

common="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT"
common+=",TRAJECTORY_HZ=$TRAJECTORY_HZ,SKIP_DATA=0,SKIP_AMORTIZER=0"

smoke_export="$common"
smoke_export+=",TRAIN_MAIN_PARAMS=12,TRIALS_PER_BLOCK=8"
smoke_export+=",VALIDATION_USERS=8,VALIDATION_TRIALS=16"
smoke_export+=",AMORTIZER_ITERATIONS=2,STEPS_PER_ITERATION=32"
smoke_export+=",BATCH_SIZE=4,SAVE_FREQUENCY=1"
smoke_export+=",AMORTIZER_NAME=PTE_SHARED_CPI_HIDDEN_CORE3_20HZ_SMOKE_$RUN_TAG"

smoke_submission="$(
  sbatch --parsable \
    --job-name=pnc_hid_c3_smoke \
    --export="$smoke_export" \
    run_hidden_cpi_core3_yonsei.slurm
)"
smoke_job="${smoke_submission%%;*}"

full_export="$common"
full_export+=",TRAIN_MAIN_PARAMS=43690,TRIALS_PER_BLOCK=64"
full_export+=",VALIDATION_USERS=100,VALIDATION_TRIALS=256"
full_export+=",AMORTIZER_ITERATIONS=200,STEPS_PER_ITERATION=2048"
full_export+=",BATCH_SIZE=16,SAVE_FREQUENCY=10"
full_export+=",AMORTIZER_NAME=PTE_SHARED_CPI_HIDDEN_CORE3_20HZ_FULL_$RUN_TAG"

full_submission="$(
  sbatch --parsable \
    --dependency="afterok:$smoke_job" \
    --job-name=pnc_hid_c3_full \
    --export="$full_export" \
    run_hidden_cpi_core3_yonsei.slurm
)"
full_job="${full_submission%%;*}"

printf '%s %s\n' "$smoke_job" "$full_job" > .last_hidden_cpi_core3_jobs

printf '%s\n' 'Submitted hidden-CPI CORE3 smoke -> full chain.'
printf 'RL model: %s (%s), environment %s\n' "$RL_MODEL" "$RL_CHECKPOINT" "$env_class"
printf '%s\n' 'Amortizer: summary-only, 20 Hz, true CPI hidden'
printf '%s\n' 'Retained curvature features (3): turn energy, early turn, late turn'
printf 'Smoke job: %s\n' "$smoke_job"
printf 'Full job:  %s (starts only if smoke succeeds)\n' "$full_job"
printf 'Status: squeue -j %s,%s\n' "$smoke_job" "$full_job"
printf 'Smoke progress: tail -F slurm-pnc-hidden-core3-%s.out\n' "$smoke_job"
printf 'Full progress:  tail -F slurm-pnc-hidden-core3-%s.out\n' "$full_job"
