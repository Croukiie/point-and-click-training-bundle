#!/usr/bin/env bash

set -euo pipefail
cd "${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
[[ -f .last_integrity_visible_cpi_pair_chain ]] || {
  printf '%s\n' 'No integrity-chain record was found.' >&2; exit 1;
}
read -r RUN_TAG RL_MODEL RL_JOB DATA_JOB SMOKE_JOB FULL_JOB PIPELINE_SEED \
  < .last_integrity_visible_cpi_pair_chain

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

last_tqdm_percent() {
  local output="$1" label="$2"
  [[ -f "$output" ]] || { printf '0'; return; }
  tr '\r' '\n' < "$output" \
    | grep -Eo "${label}: *[0-9]+%" \
    | tail -1 \
    | grep -Eo '[0-9]+%' \
    | tr -d '%' \
    || true
}

last_iteration() {
  local output="$1"
  grep -Eo 'Iter [0-9]+/[0-9]+' "$output" 2>/dev/null \
    | tail -1 \
    | awk '{split($2,a,"/"); print a[1]}' \
    || true
}

pair_values() {
  local job="$1" index="$2" total="$3"
  local output="logs/pnc-integrity-pair-${job}_${index}.out" current
  PAIR_STATE="$(state_for "${job}_${index}")"
  current="$(last_iteration "$output")"
  if [[ -z "$PAIR_STATE" ]] && grep -q 'SHARED_CPI_COMPLETE=' "$output" 2>/dev/null; then
    PAIR_STATE=COMPLETED
    current="$total"
  elif [[ -z "$PAIR_STATE" ]]; then
    PAIR_STATE=WAITING
  fi
  PAIR_PERCENT=$((100 * ${current:-0} / total))
}

render() {
  local rl_state data_state data_output data_percent
  local smoke_speed_state smoke_speed_percent smoke_core_state smoke_core_percent
  local full_speed_state full_speed_percent full_core_state full_core_percent
  local overall

  rl_state="$(state_for "$RL_JOB")"
  RL_PERCENT=0
  if [[ -z "$rl_state" ]] && [[ -f "data/pnc_model/$RL_MODEL/best/best_model.zip" ]]; then
    rl_state=COMPLETED
    RL_PERCENT=100
  elif [[ -z "$rl_state" ]]; then
    rl_state=WAITING
  fi

  data_output="logs/pnc-integrity-data-${DATA_JOB}.out"
  data_state="$(state_for "$DATA_JOB")"
  data_percent="$(last_tqdm_percent "$data_output" 'Simulating shared participants')"
  data_percent="${data_percent:-0}"
  if [[ -z "$data_state" ]] && grep -q 'INTEGRITY_DATASET_COMPLETE=' "$data_output" 2>/dev/null; then
    data_state=COMPLETED
    data_percent=100
  elif [[ -z "$data_state" ]]; then
    data_state=WAITING
  fi

  pair_values "$SMOKE_JOB" 0 2
  smoke_speed_state="$PAIR_STATE"; smoke_speed_percent="$PAIR_PERCENT"
  pair_values "$SMOKE_JOB" 1 2
  smoke_core_state="$PAIR_STATE"; smoke_core_percent="$PAIR_PERCENT"
  pair_values "$FULL_JOB" 0 200
  full_speed_state="$PAIR_STATE"; full_speed_percent="$PAIR_PERCENT"
  pair_values "$FULL_JOB" 1 200
  full_core_state="$PAIR_STATE"; full_core_percent="$PAIR_PERCENT"

  overall=$((
    (RL_PERCENT + data_percent + smoke_speed_percent + smoke_core_percent
      + full_speed_percent + full_core_percent) / 6
  ))

  printf 'Run=%s\nRL=%s\nSeed=%s\n\n' "$RUN_TAG" "$RL_MODEL" "$PIPELINE_SEED"
  printf '%-18s %-10s ' 'RL visible CPI' "$rl_state"; bar "$RL_PERCENT"; printf '\n'
  printf '%-18s %-10s ' 'Shared dataset' "$data_state"; bar "$data_percent"; printf '\n'
  printf '%-18s %-10s ' 'Smoke speedpath' "$smoke_speed_state"; bar "$smoke_speed_percent"; printf '\n'
  printf '%-18s %-10s ' 'Smoke CORE3' "$smoke_core_state"; bar "$smoke_core_percent"; printf '\n'
  printf '%-18s %-10s ' 'Full speedpath' "$full_speed_state"; bar "$full_speed_percent"; printf '\n'
  printf '%-18s %-10s ' 'Full CORE3' "$full_core_state"; bar "$full_core_percent"; printf '\n\n'
  printf '%-18s ' 'OVERALL'; bar "$overall" 36; printf '\n'
  printf '%s\n' 'Overall = unweighted mean of the six stage percentages; it is not an ETA.'
}

if [[ "${1:-}" == "--follow" ]]; then
  interval="${2:-15}"
  while true; do
    printf '\033[2J\033[H'
    render
    printf '\nRefreshing every %ss; press Ctrl+C to stop.\n' "$interval"
    sleep "$interval"
  done
else
  render
fi
