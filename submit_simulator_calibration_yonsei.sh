#!/usr/bin/env bash
# Submit the human-to-simulation calibration audit.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

MODE="${1:-smoke}"
HUMAN_TARGETS="${HUMAN_TARGETS:-calibration/human_calibration_targets.csv}"

if [[ ! -f "run_simulator_calibration_yonsei.slurm" ]]; then
  printf 'Missing run_simulator_calibration_yonsei.slurm.\n' >&2
  exit 1
fi
if [[ ! -f "$HUMAN_TARGETS" ]]; then
  printf 'Missing human target file: %s\n' "$HUMAN_TARGETS" >&2
  exit 1
fi

case "$MODE" in
  smoke)
    CALIBRATION_PROFILES="${CALIBRATION_PROFILES:-1}"
    TRIALS_PER_CPI="${TRIALS_PER_CPI:-2}"
    MINIMUM_KEEP="${MINIMUM_KEEP:-1}"
    KEEP_FRACTION="${KEEP_FRACTION:-1.0}"
    TOP_PLOT="${TOP_PLOT:-1}"
    ;;
  full)
    CALIBRATION_PROFILES="${CALIBRATION_PROFILES:-100}"
    TRIALS_PER_CPI="${TRIALS_PER_CPI:-64}"
    MINIMUM_KEEP="${MINIMUM_KEEP:-10}"
    KEEP_FRACTION="${KEEP_FRACTION:-0.10}"
    TOP_PLOT="${TOP_PLOT:-20}"
    ;;
  *)
    printf 'Usage: bash %s [smoke|full]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac

export_values="ALL,HUMAN_TARGETS=$HUMAN_TARGETS"
export_values+=",RL_MODEL=${RL_MODEL:-pnc_rl_24fhd_cpi_26210DBCT},RL_CHECKPOINT=${RL_CHECKPOINT:-best}"
export_values+=",CALIBRATION_PROFILES=$CALIBRATION_PROFILES,TRIALS_PER_CPI=$TRIALS_PER_CPI"
export_values+=",DATA_CPU=${DATA_CPU:-8},KEEP_FRACTION=$KEEP_FRACTION,MINIMUM_KEEP=$MINIMUM_KEEP"
export_values+=",TREND_WEIGHT=${TREND_WEIGHT:-2.0},TOP_PLOT=$TOP_PLOT"
if [[ -n "${SIMULATION_DATASET:-}" ]]; then
  export_values+=",SIMULATION_DATASET=$SIMULATION_DATASET"
fi

submission="$(
  sbatch --parsable \
    --job-name="pnc_calib_${MODE}" \
    --export="$export_values" \
    run_simulator_calibration_yonsei.slurm
)"
job_id="${submission%%;*}"
printf '%s\n' "$job_id" > .last_simulator_calibration_job

printf 'Submitted %s simulator calibration.\n' "$MODE"
printf 'Slurm job: %s\n' "$job_id"
printf 'Progress: bash watch_simulator_calibration_progress.sh\n'
printf 'Output:   slurm-pnc-sim-calibration-%s.out\n' "$job_id"
printf 'Errors:   slurm-pnc-sim-calibration-%s.err\n' "$job_id"
