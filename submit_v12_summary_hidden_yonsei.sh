#!/usr/bin/env bash
# Submit a 20 Hz summary-only shared-CPI amortizer for the v12 CPI-fix RL.
# The true CPI is hidden from amortizer inputs and used only after inference
# for recovery evaluation. The original RL checkpoint itself is not modified.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

MODE="${1:-smoke}"
RL_MODEL="${RL_MODEL:-pnc_rl_v12_cpifix_26204ASOS}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
TRAJECTORY_HZ="${TRAJECTORY_HZ:-20}"
FEATURE_PROFILE="${FEATURE_PROFILE:-clean7}"
RUN_TAG="${RUN_TAG:-$(date +%y%m%d_%H%M%S)}"
AMORTIZER_NAME="${AMORTIZER_NAME:-PTE_SHARED_CPI_2STEP_20HZ_V12_SUMMARY_${RUN_TAG}}"
REUSE_DATA="${REUSE_DATA:-0}"
AFTEROK_JOB_ID="${AFTEROK_JOB_ID:-}"
REQUEST_CPUS="${REQUEST_CPUS:-}"
REQUEST_MEM="${REQUEST_MEM:-}"

case "$MODE" in
  smoke)
    TRAIN_MAIN_PARAMS="${TRAIN_MAIN_PARAMS:-12}"
    TRIALS_PER_BLOCK="${TRIALS_PER_BLOCK:-8}"
    VALIDATION_USERS="${VALIDATION_USERS:-8}"
    VALIDATION_TRIALS="${VALIDATION_TRIALS:-16}"
    AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-2}"
    STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-32}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    SAVE_FREQUENCY="${SAVE_FREQUENCY:-1}"
    ;;
  full)
    TRAIN_MAIN_PARAMS="${TRAIN_MAIN_PARAMS:-43690}"
    TRIALS_PER_BLOCK="${TRIALS_PER_BLOCK:-64}"
    VALIDATION_USERS="${VALIDATION_USERS:-100}"
    VALIDATION_TRIALS="${VALIDATION_TRIALS:-256}"
    AMORTIZER_ITERATIONS="${AMORTIZER_ITERATIONS:-200}"
    STEPS_PER_ITERATION="${STEPS_PER_ITERATION:-2048}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    SAVE_FREQUENCY="${SAVE_FREQUENCY:-10}"
    ;;
  *)
    printf 'Usage: bash %s [smoke|full]\n' "${BASH_SOURCE[0]}" >&2
    exit 2
    ;;
esac

if [[ "$REUSE_DATA" != "0" && "$REUSE_DATA" != "1" ]]; then
  printf 'REUSE_DATA must be 0 or 1, got: %s\n' "$REUSE_DATA" >&2
  exit 2
fi

model_dir="data/pnc_model/$RL_MODEL"
for required in \
  "$model_dir/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip" \
  "$model_dir/env_info.yaml" \
  "$model_dir/train_config.yaml"
do
  if [[ ! -f "$required" ]]; then
    printf 'Missing RL model file: %s\n' "$required" >&2
    exit 1
  fi
done

if [[ ! -f run_shared_cpi_2step_yonsei.slurm ]]; then
  printf 'Missing launcher: run_shared_cpi_2step_yonsei.slurm\n' >&2
  exit 1
fi

env_class="$(awk -F': *' '/^env_class:/{print $2; exit}' "$model_dir/env_info.yaml")"
printf 'RL environment recorded in checkpoint: %s\n' "${env_class:-unknown}"
if [[ "$env_class" != "PnCEnvCpiHidden" ]]; then
  printf '%s\n' 'NOTE: CPI is hidden from the amortizer, but this original RL policy was trained with CPI visible.'
fi

export_values="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT"
export_values+=",VARIANT=summary-only,FEATURE_PROFILE=$FEATURE_PROFILE"
export_values+=",DATA_CPU=${DATA_CPU:-64},TRAIN_MAIN_PARAMS=$TRAIN_MAIN_PARAMS"
export_values+=",TRIALS_PER_BLOCK=$TRIALS_PER_BLOCK"
export_values+=",VALIDATION_USERS=$VALIDATION_USERS,VALIDATION_TRIALS=$VALIDATION_TRIALS"
export_values+=",TRAJECTORY_HZ=$TRAJECTORY_HZ"
export_values+=",AMORTIZER_ITERATIONS=$AMORTIZER_ITERATIONS"
export_values+=",STEPS_PER_ITERATION=$STEPS_PER_ITERATION,BATCH_SIZE=$BATCH_SIZE"
export_values+=",SAVE_FREQUENCY=$SAVE_FREQUENCY,AMORTIZER_NAME=$AMORTIZER_NAME"
export_values+=",SKIP_DATA=$REUSE_DATA,SKIP_AMORTIZER=0"

sbatch_args=(
  --parsable
  --job-name="pnc_v12_summary_${MODE}"
  --export="$export_values"
)

if [[ -n "$AFTEROK_JOB_ID" ]]; then
  if [[ ! "$AFTEROK_JOB_ID" =~ ^[0-9]+$ ]]; then
    printf 'AFTEROK_JOB_ID must be numeric, got: %s\n' "$AFTEROK_JOB_ID" >&2
    exit 2
  fi
  sbatch_args+=(--dependency="afterok:$AFTEROK_JOB_ID")
fi
if [[ -n "$REQUEST_CPUS" ]]; then
  sbatch_args+=(--cpus-per-task="$REQUEST_CPUS")
fi
if [[ -n "$REQUEST_MEM" ]]; then
  sbatch_args+=(--mem="$REQUEST_MEM")
fi

submission="$(sbatch "${sbatch_args[@]}" run_shared_cpi_2step_yonsei.slurm)"
job_id="${submission%%;*}"
printf '%s\n' "$job_id" > .last_v12_summary_hidden_job

printf 'Submitted v12 summary-only shared-CPI amortizer (%s).\n' "$MODE"
printf 'RL model: %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf 'Amortizer input: summary-only, feature profile %s, %s Hz preprocessing\n' "$FEATURE_PROFILE" "$TRAJECTORY_HZ"
printf 'True CPI input to amortizer: hidden\n'
printf 'Regenerate simulated data: %s\n' "$([[ "$REUSE_DATA" == "0" ]] && printf yes || printf no)"
printf 'Slurm job: %s\n' "$job_id"
if [[ -n "$AFTEROK_JOB_ID" ]]; then
  printf 'Starts only after successful job: %s\n' "$AFTEROK_JOB_ID"
fi
printf 'Status: squeue -j %s\n' "$job_id"
printf 'Progress: tail -F slurm-pnc-shared-cpi-%s.out\n' "$job_id"
printf 'Errors: tail -F slurm-pnc-shared-cpi-%s.err\n' "$job_id"
