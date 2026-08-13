#!/usr/bin/env bash
# Submit a matched speed/path-only smoke -> full amortizer chain.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_hidden_yonsei_30m_hidden_rl_core3_0812_173241}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
RUN_TAG="${RUN_TAG:-$(date +%y%m%d_%H%M%S)}"

for required in \
  run_hidden_cpi_speedpath_yonsei.slurm \
  src/scripts/shared_cpi_speedpath_run.py \
  "data/pnc_model/$RL_MODEL/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip"
do
  if [[ ! -f "$required" ]]; then
    printf 'Missing required file: %s\n' "$required" >&2
    exit 1
  fi
done

dataset_dir="data/amortizer_cond/dataset/${RL_MODEL}_${RL_CHECKPOINT}_20_default_012345678_9"
if [[ ! -d "$dataset_dir" ]]; then
  printf 'The matched CORE3 dataset is missing: %s\n' "$dataset_dir" >&2
  printf '%s\n' 'Do not generate a replacement silently; use the same raw trials for this ablation.' >&2
  exit 1
fi

if ! compgen -G "$dataset_dir/tr_*mp_3sp_64ep_*.npz" >/dev/null; then
  printf '%s\n' 'The full 64-trial training dataset is missing.' >&2
  exit 1
fi
if ! compgen -G "$dataset_dir/vd_100users_3_conds_256trials_*.npz" >/dev/null; then
  printf '%s\n' 'The full validation dataset is missing.' >&2
  exit 1
fi

common="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT,REUSE_DATA=1,TRAIN_MAIN_PARAMS=43690,TRIALS_PER_BLOCK=64,VALIDATION_USERS=100,VALIDATION_TRIALS=256"

smoke_submission="$(sbatch --parsable \
  --job-name=pnc_hid_sp_smoke \
  --cpus-per-task=8 \
  --export="$common,DATA_CPU=8,AMORTIZER_ITERATIONS=2,STEPS_PER_ITERATION=32,BATCH_SIZE=4,SAVE_FREQUENCY=1,AMORTIZER_NAME=PTE_SHARED_CPI_HIDDEN_SPEEDPATH_20HZ_SMOKE_$RUN_TAG" \
  run_hidden_cpi_speedpath_yonsei.slurm)"
smoke_job="${smoke_submission%%;*}"

full_submission="$(sbatch --parsable \
  --dependency="afterok:$smoke_job" \
  --job-name=pnc_hid_sp_full \
  --export="$common,DATA_CPU=16,AMORTIZER_ITERATIONS=200,STEPS_PER_ITERATION=2048,BATCH_SIZE=16,SAVE_FREQUENCY=10,AMORTIZER_NAME=PTE_SHARED_CPI_HIDDEN_SPEEDPATH_20HZ_FULL_$RUN_TAG" \
  run_hidden_cpi_speedpath_yonsei.slurm)"
full_job="${full_submission%%;*}"

printf '%s %s %s\n' "$RL_MODEL" "$smoke_job" "$full_job" \
  > .last_hidden_speedpath_jobs

printf '%s\n' 'SPEEDPATH_CHAIN_SUBMITTED'
printf 'RL model: %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf 'Reused dataset: %s\n' "$dataset_dir"
printf 'Smoke job: %s\n' "$smoke_job"
printf 'Full job:  %s (afterok:%s)\n' "$full_job" "$smoke_job"
printf 'Status: squeue -j %s,%s -o "%%.18i %%.10T %%.12M %%.40R"\n' "$smoke_job" "$full_job"
printf 'Smoke: tail -F slurm-pnc-hidden-speedpath-%s.out\n' "$smoke_job"
printf 'Full:  tail -F slurm-pnc-hidden-speedpath-%s.out\n' "$full_job"
