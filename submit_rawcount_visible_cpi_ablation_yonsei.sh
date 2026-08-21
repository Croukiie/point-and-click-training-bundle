#!/usr/bin/env bash
# One command: shared dataset -> matched smoke pair -> matched full pair.

set -euo pipefail
cd "${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs

for required in \
  run_rawcount_visible_cpi_dataset_yonsei.slurm \
  run_rawcount_visible_cpi_pair_yonsei.slurm \
  src/scripts/shared_cpi_rawcount_ablation.py \
  src/utils/raw_count_features.py
do
  [[ -f "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done

RUN_TAG="${RUN_TAG:-rawcount_visible_$(date +%y%m%d_%H%M%S)}"
RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_visible_integrity_30m_integrity_visible_260819_151300}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
PIPELINE_SEED="${PIPELINE_SEED:-20260821}"
DATA_CPU="${DATA_CPU:-64}"
REUSE_DATA="${REUSE_DATA:-0}"
CONTRACT="movement_exact20_endpoint_clickonly_rawcount125_v4"
dataset_dir="data/amortizer_cond/dataset/${RL_MODEL}_${RL_CHECKPOINT}_20_default_012345678_9_${CONTRACT}"

if [[ "$REUSE_DATA" == 1 ]]; then
  compgen -G "$dataset_dir/tr_*mp_3sp_64ep_*.npz" >/dev/null || {
    printf 'REUSE_DATA=1 but training data is missing: %s\n' "$dataset_dir" >&2; exit 1;
  }
  compgen -G "$dataset_dir/vd_100users_3_conds_256trials_*.npz" >/dev/null || {
    printf 'REUSE_DATA=1 but validation data is missing: %s\n' "$dataset_dir" >&2; exit 1;
  }
  DATA_JOB=REUSED
  smoke_dependency=()
else
  data_submission="$(sbatch --parsable \
    --export="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT,PIPELINE_SEED=$PIPELINE_SEED,DATA_CPU=$DATA_CPU" \
    run_rawcount_visible_cpi_dataset_yonsei.slurm)"
  DATA_JOB="${data_submission%%;*}"
  smoke_dependency=(--dependency="afterok:$DATA_JOB")
fi

smoke_submission="$(sbatch --parsable "${smoke_dependency[@]}" --array=0-1 \
  --job-name=pnc_rawcount_smoke \
  --export="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT,RUN_STAGE=smoke,RUN_TAG=$RUN_TAG,PIPELINE_SEED=$PIPELINE_SEED" \
  run_rawcount_visible_cpi_pair_yonsei.slurm)"
SMOKE_JOB="${smoke_submission%%;*}"

full_submission="$(sbatch --parsable --dependency="afterok:$SMOKE_JOB" --array=0-1 \
  --job-name=pnc_rawcount_full \
  --export="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT,RUN_STAGE=full,RUN_TAG=$RUN_TAG,PIPELINE_SEED=$PIPELINE_SEED" \
  run_rawcount_visible_cpi_pair_yonsei.slurm)"
FULL_JOB="${full_submission%%;*}"

printf '%s %s %s %s %s %s\n' \
  "$RUN_TAG" "$RL_MODEL" "$RL_CHECKPOINT" "$DATA_JOB" "$SMOKE_JOB" "$FULL_JOB" \
  > .last_rawcount_visible_cpi_ablation

printf '\nRAWCOUNT_VISIBLE_CPI_ABLATION_SUBMITTED\n'
printf 'RL model:       %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf 'Dataset:        %s\n' "$DATA_JOB"
printf 'Smoke pair:     %s_[0-1] (0=speedpath, 1=rawcount)\n' "$SMOKE_JOB"
printf 'Full pair:      %s_[0-1] (after both smokes)\n' "$FULL_JOB"
printf 'Progress:       bash watch_rawcount_visible_cpi_ablation.sh --follow\n'
