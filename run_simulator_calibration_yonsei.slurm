#!/usr/bin/env bash
#SBATCH --job-name=pnc_sim_calib
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-pnc-sim-calibration-%j.out
#SBATCH --error=slurm-pnc-sim-calibration-%j.err

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BUNDLE_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3.11}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

HUMAN_TARGETS="${HUMAN_TARGETS:-calibration/human_calibration_targets.csv}"
RUN_DIR="${RUN_DIR:-data/calibration/simulator_human_${SLURM_JOB_ID}}"
DATASET_DIR="$RUN_DIR/dataset"
OUTPUT_DIR="$RUN_DIR/results"
RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_26210DBCT}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
CALIBRATION_PROFILES="${CALIBRATION_PROFILES:-100}"
TRIALS_PER_CPI="${TRIALS_PER_CPI:-64}"
DATA_CPU="${DATA_CPU:-8}"
KEEP_FRACTION="${KEEP_FRACTION:-0.10}"
MINIMUM_KEEP="${MINIMUM_KEEP:-10}"
TREND_WEIGHT="${TREND_WEIGHT:-2.0}"
TOP_PLOT="${TOP_PLOT:-20}"
SIMULATION_DATASET="${SIMULATION_DATASET:-}"

if [[ ! -f "$HUMAN_TARGETS" ]]; then
  printf 'Missing human calibration targets: %s\n' "$HUMAN_TARGETS" >&2
  exit 1
fi

printf 'Python: '
"$PYTHON_BIN" --version
printf 'Human targets: %s\n' "$HUMAN_TARGETS"
printf 'RL simulator: %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf 'Run directory: %s\n' "$RUN_DIR"

if [[ -z "$SIMULATION_DATASET" ]]; then
  mkdir -p "$DATASET_DIR"
  SIMULATION_DATASET="$DATASET_DIR/fixed_cpi_profiles.npz"
  "$PYTHON_BIN" -u -m src.scripts.generate_fixed_cpi_calibration \
    --rl-model "$RL_MODEL" \
    --rl-checkpoint "$RL_CHECKPOINT" \
    --profiles "$CALIBRATION_PROFILES" \
    --trials-per-cpi "$TRIALS_PER_CPI" \
    --cpi-values 400 800 1600 \
    --num-cpu "$DATA_CPU" \
    --output-path "$SIMULATION_DATASET"
else
  printf 'Reusing explicit simulation dataset: %s\n' "$SIMULATION_DATASET"
fi

args=(
  -m src.scripts.calibrate_simulator_to_humans
  --bundle-root "$BUNDLE_DIR"
  --human-targets "$HUMAN_TARGETS"
  --output-dir "$OUTPUT_DIR"
  --keep-fraction "$KEEP_FRACTION"
  --minimum-keep "$MINIMUM_KEEP"
  --trend-weight "$TREND_WEIGHT"
  --top-plot "$TOP_PLOT"
)

args+=(--simulation-dataset "$SIMULATION_DATASET")

"$PYTHON_BIN" -u "${args[@]}"

archive="simulator_calibration_${SLURM_JOB_ID}.tar.gz"
tar -czf "$archive" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"

printf '\nCalibration complete.\n'
printf 'Results: %s\n' "$OUTPUT_DIR"
printf 'Fixed-CPI dataset: %s\n' "$SIMULATION_DATASET"
printf 'Download archive: %s\n' "$archive"
