#!/usr/bin/env bash
# One submission: corrected-radius hidden RL -> one dataset -> matched 0/3 pair.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"
mkdir -p logs

for required in \
  run_bundle_yonsei.slurm \
  run_final_hidden_cpi_dataset_yonsei.slurm \
  run_final_hidden_cpi_pair_yonsei.slurm \
  src/scripts/shared_cpi_final_pair.py
do
  [[ -f "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done

PATCHED_TASK_HASH="b25790eb02e20f942c6dc335f268721d02339db855fb23afbc60d49e53dedd48"
actual_task_hash="$(sha256sum src/agent/pnc_task.py | awk '{print $1}')"
[[ "$actual_task_hash" == "$PATCHED_TASK_HASH" ]] || {
  printf 'Target-radius correction is not installed (SHA256=%s).\n' "$actual_task_hash" >&2
  exit 1
}

grep -q 'class PnCEnvCpiHidden' src/agent/pnc_agent.py || {
  printf '%s\n' 'PnCEnvCpiHidden is missing from src/agent/pnc_agent.py' >&2
  exit 1
}

RUN_TAG="${RUN_TAG:-radiusfix_hidden_pair_$(date +%y%m%d_%H%M%S)}"
RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_hidden_radiusfix_30m_${RUN_TAG}}"
PAIR_SEED="${PAIR_SEED:-20260813}"
RL_STEPS="${RL_STEPS:-30000000}"
RL_CPUS="${RL_CPUS:-64}"

[[ ! -e "data/pnc_model/$RL_MODEL" ]] || {
  printf 'RL output already exists: data/pnc_model/%s\n' "$RL_MODEL" >&2
  exit 1
}

printf '[1/4] Submitting corrected-radius hidden-CPI RL (%s steps)\n' "$RL_STEPS"
rl_submission="$(sbatch --parsable \
  --job-name=pnc_final_hidden_rl \
  --cpus-per-task="$RL_CPUS" \
  --mem=32G \
  --export="ALL,NUM_CPU=$RL_CPUS,RUN_NAME=$RL_MODEL,ENV_CLASS_TYPE=PnCEnvCpiHidden,AGENT_TYPE=classic,PNC_TYPE=default,RL_SETTING=default,STEPS=$RL_STEPS,DEVICE=cuda,NO_TIME_TAG=1" \
  run_bundle_yonsei.slurm)"
rl_job="${rl_submission%%;*}"

printf '[2/4] Queueing one full shared dataset after RL %s\n' "$rl_job"
data_submission="$(sbatch --parsable \
  --dependency="afterok:$rl_job" \
  --job-name=pnc_final_data \
  --export="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=best,PAIR_SEED=$PAIR_SEED,DATA_CPU=64" \
  run_final_hidden_cpi_dataset_yonsei.slurm)"
data_job="${data_submission%%;*}"

printf '[3/4] Queueing matched smoke pair after dataset %s\n' "$data_job"
smoke_submission="$(sbatch --parsable \
  --dependency="afterok:$data_job" \
  --array=0-1 \
  --job-name=pnc_final_smoke \
  --export="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=best,RUN_STAGE=smoke,RUN_TAG=$RUN_TAG,PAIR_SEED=$PAIR_SEED" \
  run_final_hidden_cpi_pair_yonsei.slurm)"
smoke_job="${smoke_submission%%;*}"

printf '[4/4] Queueing matched full pair after both smokes %s\n' "$smoke_job"
full_submission="$(sbatch --parsable \
  --dependency="afterok:$smoke_job" \
  --array=0-1 \
  --job-name=pnc_final_full \
  --export="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=best,RUN_STAGE=full,RUN_TAG=$RUN_TAG,PAIR_SEED=$PAIR_SEED" \
  run_final_hidden_cpi_pair_yonsei.slurm)"
full_job="${full_submission%%;*}"

printf '%s %s %s %s %s %s %s\n' \
  "$RUN_TAG" "$RL_MODEL" "$rl_job" "$data_job" "$smoke_job" "$full_job" "$PAIR_SEED" \
  > .last_final_hidden_cpi_pair_chain

printf '\n%s\n' 'FINAL_HIDDEN_PAIR_CHAIN_SUBMITTED'
printf 'Run tag:       %s\n' "$RUN_TAG"
printf 'RL model:      %s\n' "$RL_MODEL"
printf 'RL job:        %s\n' "$rl_job"
printf 'Dataset job:   %s (afterok:%s)\n' "$data_job" "$rl_job"
printf 'Smoke pair:    %s_[0-1] (afterok:%s)\n' "$smoke_job" "$data_job"
printf 'Full pair:     %s_[0-1] (afterok:%s)\n' "$full_job" "$smoke_job"
printf 'Profiles:      0=speedpath, 1=core3\n'
printf 'Progress:      bash watch_final_hidden_cpi_pair_chain.sh\n'
