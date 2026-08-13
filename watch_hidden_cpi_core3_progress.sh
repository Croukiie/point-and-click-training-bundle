#!/usr/bin/env bash
# Show the active smoke/full stage and a compact iteration percentage bar.

set -u

cd "${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [[ ! -f .last_hidden_cpi_core3_jobs ]]; then
  printf '%s\n' 'No hidden-CPI CORE3 job record was found.' >&2
  exit 1
fi

read -r SMOKE_JOB FULL_JOB < .last_hidden_cpi_core3_jobs

while true; do
  smoke_state="$(squeue -h -j "$SMOKE_JOB" -o '%T' 2>/dev/null | head -1)"
  full_state="$(squeue -h -j "$FULL_JOB" -o '%T' 2>/dev/null | head -1)"
  smoke_log="slurm-pnc-hidden-core3-${SMOKE_JOB}.out"
  full_log="slurm-pnc-hidden-core3-${FULL_JOB}.out"

  if [[ -n "$full_state" && "$full_state" != "PENDING" ]]; then
    stage="FULL"
    state="$full_state"
    log="$full_log"
    default_total=200
  elif [[ -n "$smoke_state" ]]; then
    stage="SMOKE"
    state="$smoke_state"
    log="$smoke_log"
    default_total=2
  elif [[ -f "$full_log" ]]; then
    stage="FULL"
    state="FINISHED"
    log="$full_log"
    default_total=200
  else
    stage="WAITING_FOR_FULL"
    state="${full_state:-PENDING}"
    log="$smoke_log"
    default_total=2
  fi

  progress=""
  if [[ -f "$log" ]]; then
    progress="$(grep -oE 'Iter [0-9]+/[0-9]+' "$log" 2>/dev/null | tail -1)"
  fi
  progress_regex='Iter[[:space:]]+([0-9]+)/([0-9]+)'
  if [[ "$progress" =~ $progress_regex ]]; then
    current="${BASH_REMATCH[1]}"
    total="${BASH_REMATCH[2]}"
  else
    current=0
    total="$default_total"
  fi

  percent=$((100 * current / total))
  filled=$((percent * 30 / 100))
  empty=$((30 - filled))
  printf '\r%-7s %-10s [' "$stage" "$state"
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '-'
  printf '] %3d%% (%d/%d)' "$percent" "$current" "$total"

  if [[ -f "$full_log" ]] && grep -q 'SHARED_CPI_COMPLETE=' "$full_log"; then
    printf '\n%s\n' "$(grep 'SHARED_CPI_COMPLETE=' "$full_log" | tail -1)"
    exit 0
  fi

  if [[ -z "$smoke_state" && -z "$full_state" ]]; then
    if [[ -f "slurm-pnc-hidden-core3-${FULL_JOB}.err" ]] && \
       [[ -s "slurm-pnc-hidden-core3-${FULL_JOB}.err" ]]; then
      printf '\nFull job ended with an error. Check: slurm-pnc-hidden-core3-%s.err\n' "$FULL_JOB"
    else
      printf '\nJobs are no longer queued. Inspect both Slurm logs for the final status.\n'
    fi
    exit 1
  fi

  sleep 10
done
