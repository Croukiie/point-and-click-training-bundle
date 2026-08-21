#!/usr/bin/env bash
#SBATCH --job-name=pnc_raw_data
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/pnc-rawcount-data-%j.out
#SBATCH --error=logs/pnc-rawcount-data-%j.err

set -euo pipefail
cd "${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RL_MODEL="${RL_MODEL:?RL_MODEL is required}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
PIPELINE_SEED="${PIPELINE_SEED:-20260821}"
DATA_CPU="${DATA_CPU:-${SLURM_CPUS_PER_TASK:-64}}"
CONTRACT="movement_exact20_endpoint_clickonly_rawcount125_v4"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

model_dir="data/pnc_model/$RL_MODEL"
checkpoint="$model_dir/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip"
[[ -f "$checkpoint" ]] || { printf 'Missing RL checkpoint: %s\n' "$checkpoint" >&2; exit 1; }
env_class="$(awk -F': *' '/^env_class:/{print $2; exit}' "$model_dir/env_info.yaml" 2>/dev/null | tr -d '\r')"
[[ "$env_class" == PnCEnvDefault ]] || {
  printf 'Expected CPI-visible PnCEnvDefault, found: <%s>\n' "${env_class:-missing}" >&2
  exit 1
}

"$PYTHON_BIN" -u -m src.scripts.shared_cpi_rawcount_ablation \
  --rl-model "$RL_MODEL" \
  --rl-checkpoint "$RL_CHECKPOINT" \
  --variant summary-only \
  --feature-profile rawcount \
  --data-cpu "$DATA_CPU" \
  --train-main-params 43690 \
  --blocks-per-participant 3 \
  --trials-per-block 64 \
  --validation-users 100 \
  --validation-trials 256 \
  --trajectory-downsample 20 \
  --amortizer-name RAWCOUNT_DATASET_ONLY \
  --amortizer-iterations 1 \
  --steps-per-iteration 1 \
  --batch-size 1 \
  --save-frequency 1 \
  --device cuda \
  --seed "$PIPELINE_SEED" \
  --skip-amortizer

dataset_dir="data/amortizer_cond/dataset/${RL_MODEL}_${RL_CHECKPOINT}_20_default_012345678_9_${CONTRACT}"
compgen -G "$dataset_dir/tr_*mp_3sp_64ep_*.npz" >/dev/null || {
  printf 'Training dataset was not created: %s\n' "$dataset_dir" >&2; exit 1;
}
compgen -G "$dataset_dir/vd_100users_3_conds_256trials_*.npz" >/dev/null || {
  printf 'Validation dataset was not created: %s\n' "$dataset_dir" >&2; exit 1;
}
printf 'RAWCOUNT_DATASET_COMPLETE=%s\n' "$dataset_dir"
