#!/usr/bin/env bash

set -euo pipefail
cd "${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [[ ! -f .last_hidden_speedpath_jobs ]]; then
  printf '%s\n' 'No hidden speed/path job record was found.' >&2
  exit 1
fi

read -r RL_MODEL SMOKE_JOB FULL_JOB < .last_hidden_speedpath_jobs

job_state() {
  local job_id="$1"
  local state
  state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -1 || true)"
  if [[ -n "$state" ]]; then
    printf '%s' "$state"
  elif grep -q 'SHARED_CPI_COMPLETE=' "slurm-pnc-hidden-speedpath-${job_id}.out" 2>/dev/null; then
    printf '%s' 'COMPLETED'
  elif [[ -s "slurm-pnc-hidden-speedpath-${job_id}.err" ]]; then
    printf '%s' 'FAILED_OR_ENDED'
  else
    printf '%s' 'NOT_IN_QUEUE'
  fi
}

progress_line() {
  local label="$1"
  local job_id="$2"
  local state current total percent filled empty bar
  state="$(job_state "$job_id")"
  total="$(grep -Eo 'SHARED_CPI_PROGRESS_TOTAL=[0-9]+' "slurm-pnc-hidden-speedpath-${job_id}.out" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
  current="$(grep -Eo 'Iter [0-9]+/[0-9]+' "slurm-pnc-hidden-speedpath-${job_id}.out" 2>/dev/null | tail -1 | awk '{split($2,a,"/"); print a[1]}' || true)"
  total="${total:-0}"
  current="${current:-0}"
  if [[ "$state" == COMPLETED ]]; then current="$total"; fi
  if (( total > 0 )); then percent=$((100 * current / total)); else percent=0; fi
  filled=$((30 * percent / 100))
  empty=$((30 - filled))
  printf -v bar '%*s' "$filled" ''; bar="${bar// /#}"
  printf -v empty '%*s' "$empty" ''; empty="${empty// /-}"
  printf '%-7s %-16s [%s%s] %3d%% (%d/%d) job=%s\n' \
    "$label" "$state" "$bar" "$empty" "$percent" "$current" "$total" "$job_id"
}

printf 'RL model: %s\n' "$RL_MODEL"
progress_line SMOKE "$SMOKE_JOB"
progress_line FULL "$FULL_JOB"
