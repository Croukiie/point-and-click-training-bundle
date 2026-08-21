#!/usr/bin/env bash
# Show measured per-iteration durations from an integrity-chain amortizer log.

set -euo pipefail
cd "${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

stage="${1:-full}"
profile="${2:-speedpath}"
follow="${3:-}"
interval="${4:-15}"

[[ "$stage" == smoke || "$stage" == full ]] || {
  printf '%s\n' 'Stage must be smoke or full.' >&2; exit 2;
}
[[ "$profile" == speedpath || "$profile" == core3 ]] || {
  printf '%s\n' 'Profile must be speedpath or core3.' >&2; exit 2;
}
[[ -f .last_integrity_visible_cpi_pair_chain ]] || {
  printf '%s\n' 'No integrity-chain record was found.' >&2; exit 1;
}

read -r RUN_TAG RL_MODEL RL_JOB DATA_JOB SMOKE_JOB FULL_JOB PIPELINE_SEED \
  < .last_integrity_visible_cpi_pair_chain
job="$SMOKE_JOB"
[[ "$stage" == full ]] && job="$FULL_JOB"
index=0
[[ "$profile" == core3 ]] && index=1
log="logs/pnc-integrity-pair-${job}_${index}.out"

render() {
  local state
  state="$(squeue -h -j "${job}_${index}" -o '%T' 2>/dev/null | head -1 || true)"
  [[ -n "$state" ]] || state=NOT_IN_QUEUE
  printf 'Run=%s  Stage=%s  Profile=%s  Job=%s_%s  State=%s\n' \
    "$RUN_TAG" "$stage" "$profile" "$job" "$index" "$state"
  printf 'Log=%s\n\n' "$log"

  .venv/bin/python - "$log" <<'PY'
from __future__ import annotations

from pathlib import Path
import re
import statistics
import sys


def seconds(text: str) -> float:
    values = [float(value) for value in text.split(":")]
    total = 0.0
    for value in values:
        total = total * 60.0 + value
    return total


def duration(value: float | None) -> str:
    if value is None:
        return "-"
    value = max(0, int(round(value)))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


path = Path(sys.argv[1])
if not path.exists():
    print("No amortizer log exists yet; the job has not started training.")
    raise SystemExit(0)

text = path.read_bytes().decode("utf-8", errors="replace").replace("\r", "\n")
complete_pattern = re.compile(
    r"Iter\s+(\d+)/(\d+):\s+100%.*?\[([0-9:]+)<[^\]]*\]"
)
partial_pattern = re.compile(r"Iter\s+(\d+)/(\d+):\s+(\d+)%")
recovery_pattern = re.compile(r"parameter recovery \(simulated\) \(([0-9.]+)s\)")

training: dict[int, tuple[int, float]] = {}
recovery: dict[int, float] = {}
last_complete: int | None = None
latest_partial: tuple[int, int, int] | None = None

for line in text.splitlines():
    completed = complete_pattern.search(line)
    if completed:
        iteration, total = map(int, completed.group(1, 2))
        training[iteration] = (total, seconds(completed.group(3)))
        last_complete = iteration
        latest_partial = (iteration, total, 100)
        continue
    partial = partial_pattern.search(line)
    if partial:
        latest_partial = tuple(map(int, partial.groups()))
    recovered = recovery_pattern.search(line)
    if recovered and last_complete is not None:
        recovery[last_complete] = recovery.get(last_complete, 0.0) + float(
            recovered.group(1)
        )

if not training:
    if latest_partial:
        print(
            f"Current iteration: {latest_partial[0]}/{latest_partial[1]} "
            f"({latest_partial[2]}%)"
        )
    else:
        print("No completed amortizer iteration yet.")
    raise SystemExit(0)

iterations = sorted(training)
print(f"{'Iteration':>10}  {'training':>10}  {'recovery':>10}  {'observed total':>14}")
for iteration in iterations[-10:]:
    total, train_s = training[iteration]
    recovery_s = recovery.get(iteration)
    observed = train_s + recovery_s if recovery_s is not None else train_s
    print(
        f"{iteration:>4}/{total:<4}  {duration(train_s):>10}  "
        f"{duration(recovery_s):>10}  {duration(observed):>14}"
    )

recent = iterations[-10:]
average_training = statistics.mean(training[item][1] for item in recent)
last = iterations[-1]
total = training[last][0]
remaining = max(total - last, 0)
print()
print(f"Average training time, last {len(recent)} iteration(s): {duration(average_training)}")
print(f"Approximate remaining training-loop time: {duration(average_training * remaining)}")
if latest_partial and latest_partial[2] < 100:
    print(
        f"Current iteration: {latest_partial[0]}/{latest_partial[1]} "
        f"({latest_partial[2]}%)"
    )
print("ETA excludes future checkpoint saving/recovery overhead and Slurm waiting time.")
PY
}

if [[ "$follow" == --follow ]]; then
  while true; do
    printf '\033[2J\033[H'
    render
    printf '\nRefreshing every %ss; press Ctrl+C to stop.\n' "$interval"
    sleep "$interval"
  done
else
  render
fi
