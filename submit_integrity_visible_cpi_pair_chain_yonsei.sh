#!/usr/bin/env bash
# Corrected CPI-visible RL -> versioned dataset -> matched smoke -> matched full.

set -euo pipefail
cd "${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs

for required in run_bundle_yonsei.slurm \
  run_integrity_visible_cpi_dataset_yonsei.slurm \
  run_integrity_visible_cpi_pair_yonsei.slurm \
  src/scripts/shared_cpi_final_visible_pair.py
do
  [[ -f "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done

RUN_TAG="${RUN_TAG:-integrity_visible_$(date +%y%m%d_%H%M%S)}"
RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_visible_integrity_30m_${RUN_TAG}}"
PIPELINE_SEED="${PIPELINE_SEED:-20260819}"
RL_STEPS="${RL_STEPS:-30000000}"
RL_CPUS="${RL_CPUS:-64}"

[[ ! -e "data/pnc_model/$RL_MODEL" ]] || {
  printf 'RL output already exists: data/pnc_model/%s\n' "$RL_MODEL" >&2; exit 1;
}

rl_submission="$(sbatch --parsable \
  --job-name=pnc_integrity_rl \
  --cpus-per-task="$RL_CPUS" \
  --mem=32G \
  --export="ALL,NUM_CPU=$RL_CPUS,RUN_NAME=$RL_MODEL,ENV_CLASS_TYPE=PnCEnvDefault,AGENT_TYPE=classic,PNC_TYPE=default,RL_SETTING=default,STEPS=$RL_STEPS,DEVICE=cuda,NO_TIME_TAG=1,SEED=$PIPELINE_SEED" \
  run_bundle_yonsei.slurm)"
RL_JOB="${rl_submission%%;*}"

data_submission="$(sbatch --parsable --dependency="afterok:$RL_JOB" \
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
  > .last_integrity_visible_cpi_pair_chain

printf '\nINTEGRITY_VISIBLE_CPI_CHAIN_SUBMITTED\n'
printf 'RL model: %s\nRL job: %s\nDataset job: %s\nSmoke pair: %s_[0-1]\nFull pair: %s_[0-1]\n' \
  "$RL_MODEL" "$RL_JOB" "$DATA_JOB" "$SMOKE_JOB" "$FULL_JOB"
printf 'Progress: bash watch_integrity_visible_cpi_pair_chain.sh\n'

