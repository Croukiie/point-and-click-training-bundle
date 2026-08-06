#!/usr/bin/env bash

set -u

if [[ $# -lt 1 ]]; then
  printf 'Usage: bash %s JOBID [ITERATIONS] [STEPS_PER_ITERATION]\n' "$0" >&2
  exit 2
fi

job_id="$1"
total_iterations="${2:-200}"
steps_per_iteration="${3:-2048}"
out_file="slurm-pnc-curvature-only-${job_id}.out"
err_file="slurm-pnc-curvature-only-${job_id}.err"
total_steps=$((total_iterations * steps_per_iteration))
bar_width=40

printf 'Job %s — Ctrl+C stops only this display, not the Slurm job.\n' "$job_id"

while true; do
  state="$(squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -1)"
  iteration=0
  step=0

  if [[ -f "$err_file" ]]; then
    progress_line="$(tail -c 250000 "$err_file" 2>/dev/null \
      | tr '\r' '\n' \
      | grep -E 'Iter [0-9]+:.*[0-9]+/[0-9]+' \
      | tail -1 || true)"

    if [[ -n "$progress_line" ]]; then
      iteration="$(sed -nE 's/.*Iter ([0-9]+):.*/\1/p' <<<"$progress_line")"
      fraction="$(sed -nE 's#.* ([0-9]+)/([0-9]+) .*#\1 \2#p' <<<"$progress_line")"
      if [[ -n "$fraction" ]]; then
        read -r step _ <<<"$fraction"
      fi
    fi
  fi

  if [[ -f "$out_file" ]] && grep -q '\[ Training Done \]' "$out_file"; then
    iteration="$total_iterations"
    step="$steps_per_iteration"
  fi

  if [[ "$iteration" =~ ^[0-9]+$ && "$step" =~ ^[0-9]+$ && "$iteration" -gt 0 ]]; then
    completed_steps=$(( (iteration - 1) * steps_per_iteration + step ))
  else
    completed_steps=0
  fi
  (( completed_steps > total_steps )) && completed_steps="$total_steps"

  filled=$((completed_steps * bar_width / total_steps))
  empty=$((bar_width - filled))
  printf -v filled_bar '%*s' "$filled" ''
  printf -v empty_bar '%*s' "$empty" ''
  filled_bar="${filled_bar// /#}"
  empty_bar="${empty_bar// /-}"
  percent="$(awk -v done="$completed_steps" -v total="$total_steps" \
    'BEGIN { printf "%.2f", (total > 0 ? 100 * done / total : 0) }')"

  printf '\r[%s%s] %6s%% | iteration %3d/%d | step %4d/%d | %-10s' \
    "$filled_bar" "$empty_bar" "$percent" \
    "$iteration" "$total_iterations" "$step" "$steps_per_iteration" \
    "${state:-finished}"

  if [[ -f "$out_file" ]] && grep -q 'CURVATURE_ONLY_COMPLETE=' "$out_file"; then
    printf '\nCompleted successfully.\n'
    grep 'CURVATURE_ONLY_COMPLETE=' "$out_file" | tail -1
    exit 0
  fi

  if [[ -z "$state" && ( -f "$out_file" || -f "$err_file" ) ]]; then
    printf '\nThe job is no longer in the queue. Last output:\n'
    [[ -f "$out_file" ]] && tail -n 15 "$out_file"
    [[ -f "$err_file" ]] && tail -n 15 "$err_file"
    exit 1
  fi

  sleep 5
done
