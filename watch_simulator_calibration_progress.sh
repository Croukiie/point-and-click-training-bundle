#!/usr/bin/env bash
# Follow calibration stages and their percentage bar.

set -euo pipefail

if [[ $# -gt 0 ]]; then
  job_id="$1"
  output="slurm-pnc-sim-calibration-${job_id}.out"
elif [[ -s .last_simulator_calibration_job ]]; then
  job_id="$(tr -d '[:space:]' < .last_simulator_calibration_job)"
  output="slurm-pnc-sim-calibration-${job_id}.out"
else
  output="$(ls -t slurm-pnc-sim-calibration-*.out 2>/dev/null | head -1 || true)"
fi

if [[ -z "${output:-}" ]]; then
  printf 'No simulator-calibration Slurm output was found.\n' >&2
  exit 1
fi

if [[ ! -f "$output" ]]; then
  printf 'Waiting for Slurm to create %s. The job may still be pending.\n' "$output"
  if [[ -n "${job_id:-}" ]]; then
    squeue -j "$job_id" -o '%.18i %.9P %.16j %.2t %.10M %.30R' || true
  fi
  exit 0
fi

printf 'Watching %s (Ctrl+C stops watching, not the job).\n' "$output"
tail -F "$output"
