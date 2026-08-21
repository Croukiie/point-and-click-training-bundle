#!/usr/bin/env bash
# Reuse a completed integrity RL after a dataset-stage failure.

set -euo pipefail
cd "${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

record=.last_integrity_visible_cpi_pair_chain
[[ -f "$record" ]] || { printf '%s\n' 'No integrity-chain record was found.' >&2; exit 1; }
read -r RUN_TAG RL_MODEL RL_JOB OLD_DATA_JOB OLD_SMOKE_JOB OLD_FULL_JOB PIPELINE_SEED \
  < "$record"

checkpoint="data/pnc_model/$RL_MODEL/best/best_model.zip"
[[ -f "$checkpoint" ]] || { printf 'Missing completed RL checkpoint: %s\n' "$checkpoint" >&2; exit 1; }

active_old="$(squeue -h -j "$OLD_DATA_JOB,$OLD_SMOKE_JOB,$OLD_FULL_JOB" -o '%A' 2>/dev/null || true)"
if [[ -n "$active_old" ]]; then
  printf '%s\n' 'Old downstream jobs are still in the queue.' >&2
  printf 'Cancel them first with: scancel %s %s %s\n' \
    "$OLD_DATA_JOB" "$OLD_SMOKE_JOB" "$OLD_FULL_JOB" >&2
  exit 3
fi

cp "$record" "${record}.before_dataset_retry_$(date +%Y%m%d_%H%M%S)"

data_submission="$(sbatch --parsable \
  --job-name=pnc_integrity_data \
  --export="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=best,PIPELINE_SEED=$PIPELINE_SEED,DATA_CPU=64" \
  run_integrity_visible_cpi_dataset_yonsei.slurm)"
DATA_JOB="${data_submission%%;*}"

smoke_submission="$(sbatch --parsable --dependency="afterok:$DATA_JOB" --array=0-1 \
  --job-name=pnc_integrity_smoke \
  --export="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=best,RUN_STAGE=smoke,RUN_TAG=$RUN_TAG,PIPELINE_SEED=$PIPELINE_SEED" \
  run_integrity_visible_cpi_pair_yonsei.slurm)"
SMOKE_JOB="${smoke_submission%%;*}"

full_submission="$(sbatch --parsable --dependency="afterok:$SMOKE_JOB" --array=0-1 \
  --job-name=pnc_integrity_full \
  --export="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=best,RUN_STAGE=full,RUN_TAG=$RUN_TAG,PIPELINE_SEED=$PIPELINE_SEED" \
  run_integrity_visible_cpi_pair_yonsei.slurm)"
FULL_JOB="${full_submission%%;*}"

printf '%s %s %s %s %s %s %s\n' \
  "$RUN_TAG" "$RL_MODEL" "$RL_JOB" "$DATA_JOB" "$SMOKE_JOB" "$FULL_JOB" "$PIPELINE_SEED" \
  > "$record"

printf '\nINTEGRITY_DATASET_AND_AMORTIZERS_RESUBMITTED\n'
printf 'Reused RL: %s (job %s)\n' "$RL_MODEL" "$RL_JOB"
printf 'Dataset job: %s\nSmoke pair: %s_[0-1]\nFull pair: %s_[0-1]\n' \
  "$DATA_JOB" "$SMOKE_JOB" "$FULL_JOB"
printf '%s\n' 'Progress: bash watch_integrity_visible_cpi_pair_chain.sh'
