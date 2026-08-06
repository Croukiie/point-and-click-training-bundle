#!/usr/bin/env bash
# Show the latest shared-CPI training progress, including tqdm percentages.

set -euo pipefail

if [[ $# -gt 0 ]]; then
  job_id="$1"
  output="slurm-pnc-shared-cpi-${job_id}.out"
else
  output="$(ls -t slurm-pnc-shared-cpi-*.out 2>/dev/null | head -1 || true)"
fi

if [[ -z "${output:-}" || ! -f "$output" ]]; then
  printf 'No shared-CPI Slurm output was found.\n' >&2
  exit 1
fi

printf 'Watching %s (Ctrl+C stops watching, not the job).\n' "$output"
tail -F "$output"
