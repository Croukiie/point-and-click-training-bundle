#!/usr/bin/env bash

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"
mkdir -p logs

RUN_MODE="${1:-smoke}"
case "$RUN_MODE" in
  smoke|diagnostic|full) ;;
  *) printf 'Usage: %s [smoke|diagnostic|full]\n' "$0" >&2; exit 2 ;;
esac

RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_hidden_yonsei_30m_hidden_rl_core3_0812_173241}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
RUN_TAG="${RUN_TAG:-$(date +%y%m%d_%H%M%S)}"
ABLATION_SEED="${ABLATION_SEED:-20260813}"

submission="$(sbatch --parsable \
  --array=0-3 \
  --export="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT,RUN_MODE=$RUN_MODE,RUN_TAG=$RUN_TAG,ABLATION_SEED=$ABLATION_SEED" \
  run_shared_cpi_ablation_matrix_yonsei.slurm)"
job_id="${submission%%;*}"

printf '%s %s %s %s %s\n' \
  "$job_id" "$RUN_MODE" "$RUN_TAG" "$RL_MODEL" "$ABLATION_SEED" \
  > .last_shared_cpi_ablation_matrix

printf '%s\n' 'ABLATION_MATRIX_SUBMITTED'
printf 'Array job: %s_[0-3]\n' "$job_id"
printf 'Mode: %s\n' "$RUN_MODE"
printf 'Profiles: 0=speedpath, 1=core3, 2=clean7, 3=full10\n'
printf 'Status: squeue -j %s\n' "$job_id"
printf 'Progress: bash watch_shared_cpi_ablation_matrix.sh\n'
