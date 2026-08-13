#!/usr/bin/env bash

set -euo pipefail
cd "${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

[[ -f .last_final_hidden_cpi_pair_chain ]] || {
  printf '%s\n' 'No final hidden-CPI pair chain record was found.' >&2
  exit 1
}

read -r RUN_TAG RL_MODEL RL_JOB DATA_JOB SMOKE_JOB FULL_JOB PAIR_SEED \
  < .last_final_hidden_cpi_pair_chain

state_for() {
  local task="$1"
  squeue -h -j "$task" -o '%T' 2>/dev/null | head -1 || true
}

bar() {
  local current="$1" total="$2" width=24 filled empty percent filled_bar empty_bar
  if (( total <= 0 )); then printf '[------------------------]   0%%'; return; fi
  percent=$((100 * current / total))
  filled=$((width * current / total))
  empty=$((width - filled))
  filled_bar="$(printf '%*s' "$filled" '' | tr ' ' '#')"
  empty_bar="$(printf '%*s' "$empty" '' | tr ' ' '-')"
  printf '[%s%s] %3d%%' "$filled_bar" "$empty_bar" "$percent"
}

print_pair_task() {
  local label="$1" job="$2" index="$3" total="$4"
  local task="${job}_${index}" output="logs/pnc-final-pair-${job}_${index}.out"
  local error="logs/pnc-final-pair-${job}_${index}.err" state current
  state="$(state_for "$task")"
  current="$(grep -Eo 'Iter [0-9]+/[0-9]+' "$output" 2>/dev/null | tail -1 | awk '{split($2,a,"/"); print a[1]}' || true)"
  if [[ -z "$state" ]] && grep -q 'SHARED_CPI_COMPLETE=' "$output" 2>/dev/null; then
    state=COMPLETED
    current="$total"
  elif [[ -z "$state" && -s "$error" ]]; then
    state=STOPPED
  elif [[ -z "$state" ]]; then
    state=WAITING
  fi
  printf '%-18s %-10s ' "$label" "$state"
  bar "${current:-0}" "$total"
  printf ' (%s/%s)\n' "${current:-0}" "$total"
}

rl_state="$(state_for "$RL_JOB")"
if [[ -z "$rl_state" && -f "data/pnc_model/$RL_MODEL/best/best_model.zip" ]]; then
  rl_state=COMPLETED
elif [[ -z "$rl_state" ]]; then
  rl_state=WAITING
fi

data_state="$(state_for "$DATA_JOB")"
if [[ -z "$data_state" ]] && grep -q 'FINAL_DATASET_COMPLETE=' "logs/pnc-final-data-${DATA_JOB}.out" 2>/dev/null; then
  data_state=COMPLETED
elif [[ -z "$data_state" && -s "logs/pnc-final-data-${DATA_JOB}.err" ]]; then
  data_state=STOPPED
elif [[ -z "$data_state" ]]; then
  data_state=WAITING
fi

printf 'Run=%s\nRL=%s\nSeed=%s\n\n' "$RUN_TAG" "$RL_MODEL" "$PAIR_SEED"
printf '%-18s %s\n' 'RL (30M)' "$rl_state"
printf '%-18s %s\n' 'Shared dataset' "$data_state"
print_pair_task 'Smoke speedpath' "$SMOKE_JOB" 0 2
print_pair_task 'Smoke CORE3' "$SMOKE_JOB" 1 2
print_pair_task 'Full speedpath' "$FULL_JOB" 0 200
print_pair_task 'Full CORE3' "$FULL_JOB" 1 200

printf '\nRaw status: squeue -j %s,%s,%s,%s\n' "$RL_JOB" "$DATA_JOB" "$SMOKE_JOB" "$FULL_JOB"
