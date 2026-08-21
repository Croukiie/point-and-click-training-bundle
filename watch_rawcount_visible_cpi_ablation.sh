#!/usr/bin/env bash

set -euo pipefail
cd "${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[[ -f .last_rawcount_visible_cpi_ablation ]] || {
  printf '%s\n' 'No raw-count ablation record was found.' >&2; exit 1;
}
read -r RUN_TAG RL_MODEL RL_CHECKPOINT DATA_JOB SMOKE_JOB FULL_JOB \
  < .last_rawcount_visible_cpi_ablation

state_for() { squeue -h -j "$1" -o '%T' 2>/dev/null | head -1 || true; }
bar() {
  local percent="${1:-0}" width="${2:-24}" filled
  (( percent < 0 )) && percent=0
  (( percent > 100 )) && percent=100
  filled=$((width * percent / 100))
  printf '[%s%s] %3d%%' \
    "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
    "$(printf '%*s' "$((width-filled))" '' | tr ' ' '-')" \
    "$percent"
}
last_iteration() {
  grep -Eo 'Iter [0-9]+/[0-9]+' "$1" 2>/dev/null \
    | tail -1 | awk '{split($2,a,"/"); print a[1]}' || true
}
last_dataset_percent() {
  [[ -f "$1" ]] || { printf '0'; return; }
  tr '\r' '\n' < "$1" \
    | grep -Eo 'Simulating shared participants: *[0-9]+%' \
    | tail -1 | grep -Eo '[0-9]+%' | tr -d '%' || true
}
pair_values() {
  local job="$1" index="$2" total="$3"
  local output="logs/pnc-rawcount-pair-${job}_${index}.out" current
  PAIR_STATE="$(state_for "${job}_${index}")"
  current="$(last_iteration "$output")"
  if [[ -z "$PAIR_STATE" ]] && grep -q 'SHARED_CPI_COMPLETE=' "$output" 2>/dev/null; then
    PAIR_STATE=COMPLETED; current="$total"
  elif [[ -z "$PAIR_STATE" ]]; then
    PAIR_STATE=WAITING
  fi
  PAIR_PERCENT=$((100 * ${current:-0} / total))
}
render() {
  local data_state data_percent data_output overall
  if [[ "$DATA_JOB" == REUSED ]]; then
    data_state=REUSED; data_percent=100
  else
    data_state="$(state_for "$DATA_JOB")"
    data_output="logs/pnc-rawcount-data-${DATA_JOB}.out"
    data_percent="$(last_dataset_percent "$data_output")"; data_percent="${data_percent:-0}"
    if [[ -z "$data_state" ]] && grep -q 'RAWCOUNT_DATASET_COMPLETE=' "$data_output" 2>/dev/null; then
      data_state=COMPLETED; data_percent=100
    elif [[ -z "$data_state" ]]; then
      data_state=WAITING
    fi
  fi

  pair_values "$SMOKE_JOB" 0 2
  local ss="$PAIR_STATE" sp="$PAIR_PERCENT"
  pair_values "$SMOKE_JOB" 1 2
  local rs="$PAIR_STATE" rp="$PAIR_PERCENT"
  pair_values "$FULL_JOB" 0 200
  local fs="$PAIR_STATE" fp="$PAIR_PERCENT"
  pair_values "$FULL_JOB" 1 200
  local frs="$PAIR_STATE" frp="$PAIR_PERCENT"
  overall=$(((data_percent + sp + rp + fp + frp) / 5))

  printf 'Run=%s\nRL=%s (%s)\n\n' "$RUN_TAG" "$RL_MODEL" "$RL_CHECKPOINT"
  printf '%-18s %-10s ' 'Shared dataset' "$data_state"; bar "$data_percent"; printf '\n'
  printf '%-18s %-10s ' 'Smoke speedpath' "$ss"; bar "$sp"; printf '\n'
  printf '%-18s %-10s ' 'Smoke rawcount' "$rs"; bar "$rp"; printf '\n'
  printf '%-18s %-10s ' 'Full speedpath' "$fs"; bar "$fp"; printf '\n'
  printf '%-18s %-10s ' 'Full rawcount' "$frs"; bar "$frp"; printf '\n\n'
  printf '%-18s ' 'OVERALL'; bar "$overall" 36; printf '\n'
}

if [[ "${1:-}" == --follow ]]; then
  interval="${2:-15}"
  while true; do
    printf '\033[2J\033[H'; render
    printf '\nRefreshing every %ss; press Ctrl+C to stop.\n' "$interval"
    sleep "$interval"
  done
else
  render
fi
