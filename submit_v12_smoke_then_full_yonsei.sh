#!/usr/bin/env bash
# Submit a lightweight smoke test and immediately queue the full run behind it.
# The full run starts only if the smoke job exits successfully.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

if [[ ! -f submit_v12_summary_hidden_yonsei.sh ]]; then
  printf 'Missing submit_v12_summary_hidden_yonsei.sh\n' >&2
  exit 1
fi

RUN_BASE="${RUN_BASE:-$(date +%y%m%d_%H%M%S)}"
FEATURE_PROFILE="${FEATURE_PROFILE:-clean7}"
SMOKE_CPUS="${SMOKE_CPUS:-8}"
SMOKE_MEM="${SMOKE_MEM:-32G}"
SMOKE_DATA_CPU="${SMOKE_DATA_CPU:-8}"

printf '[1/2] Submitting lightweight smoke test...\n'
FEATURE_PROFILE="$FEATURE_PROFILE" \
REQUEST_CPUS="$SMOKE_CPUS" \
REQUEST_MEM="$SMOKE_MEM" \
DATA_CPU="$SMOKE_DATA_CPU" \
REUSE_DATA=0 \
RUN_TAG="${RUN_BASE}_smoke" \
  bash submit_v12_summary_hidden_yonsei.sh smoke

smoke_job="$(<.last_v12_summary_hidden_job)"
if [[ ! "$smoke_job" =~ ^[0-9]+$ ]]; then
  printf 'Invalid smoke job id: %s\n' "$smoke_job" >&2
  exit 1
fi

printf '[2/2] Queueing full run after successful smoke job %s...\n' "$smoke_job"
FEATURE_PROFILE="$FEATURE_PROFILE" \
AFTEROK_JOB_ID="$smoke_job" \
REUSE_DATA=0 \
RUN_TAG="${RUN_BASE}_full" \
  bash submit_v12_summary_hidden_yonsei.sh full

full_job="$(<.last_v12_summary_hidden_job)"
if [[ ! "$full_job" =~ ^[0-9]+$ ]]; then
  printf 'Invalid full job id: %s\n' "$full_job" >&2
  exit 1
fi

printf '%s %s\n' "$smoke_job" "$full_job" > .last_v12_smoke_full_jobs

printf '\nQueued smoke -> full chain.\n'
printf 'Smoke job: %s (8 CPU/32G by default)\n' "$smoke_job"
printf 'Full job:  %s (afterok:%s)\n' "$full_job" "$smoke_job"
printf 'Status: squeue -j %s,%s -o "%%.18i %%.10T %%.12M %%.40R"\n' "$smoke_job" "$full_job"
printf 'Smoke log: tail -F slurm-pnc-shared-cpi-%s.out\n' "$smoke_job"
printf 'Full log:  tail -F slurm-pnc-shared-cpi-%s.out\n' "$full_job"
