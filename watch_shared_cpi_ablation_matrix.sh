#!/usr/bin/env bash

set -euo pipefail
cd "${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

read -r JOB_ID RUN_MODE RUN_TAG RL_MODEL ABLATION_SEED \
  < .last_shared_cpi_ablation_matrix
PROFILES=(speedpath core3 clean7 full10)

printf 'RL=%s mode=%s seed=%s\n' "$RL_MODEL" "$RUN_MODE" "$ABLATION_SEED"
for index in 0 1 2 3; do
  task="${JOB_ID}_${index}"
  state="$(squeue -h -j "$task" -o '%T' 2>/dev/null | head -1 || true)"
  output="logs/pnc-cpi-ablation-${JOB_ID}_${index}.out"
  current="$(grep -Eo 'Iter [0-9]+/[0-9]+' "$output" 2>/dev/null | tail -1 || true)"
  if [[ -z "$state" ]] && grep -q 'SHARED_CPI_COMPLETE=' "$output" 2>/dev/null; then
    state=COMPLETED
  fi
  printf '%-10s %-12s %s\n' "${PROFILES[$index]}" "${state:-NOT_IN_QUEUE}" "${current:-waiting}"
done
