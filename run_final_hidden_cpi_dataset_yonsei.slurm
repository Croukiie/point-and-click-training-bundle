#!/usr/bin/env bash
#SBATCH --job-name=pnc_final_data
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/pnc-final-data-%j.out
#SBATCH --error=logs/pnc-final-data-%j.err

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BUNDLE_DIR"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RL_MODEL="${RL_MODEL:?RL_MODEL is required}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
PAIR_SEED="${PAIR_SEED:-20260813}"
DATA_CPU="${DATA_CPU:-${SLURM_CPUS_PER_TASK:-64}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

model_dir="data/pnc_model/$RL_MODEL"
checkpoint="$model_dir/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip"
[[ -f "$checkpoint" ]] || { printf 'Missing RL checkpoint: %s\n' "$checkpoint" >&2; exit 1; }

env_class="$(awk -F': *' '/^env_class:/{print $2; exit}' "$model_dir/env_info.yaml" 2>/dev/null | tr -d '\r')"
[[ "$env_class" == PnCEnvCpiHidden ]] || {
  printf 'Expected PnCEnvCpiHidden, found: <%s>\n' "${env_class:-missing}" >&2
  exit 1
}

"$PYTHON_BIN" - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this allocation")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

printf 'FINAL_DATASET_RL=%s\n' "$RL_MODEL"
printf 'FINAL_DATASET_SEED=%s\n' "$PAIR_SEED"
printf '%s\n' 'FINAL_DATASET_DESIGN=20Hz,summary-only,9-shared+1-block-CPI,3-blocks,64-trials'

"$PYTHON_BIN" -u -m src.scripts.shared_cpi_final_pair \
  --rl-model "$RL_MODEL" \
  --rl-checkpoint "$RL_CHECKPOINT" \
  --variant summary-only \
  --feature-profile core3 \
  --data-cpu "$DATA_CPU" \
  --train-main-params 43690 \
  --blocks-per-participant 3 \
  --trials-per-block 64 \
  --validation-users 100 \
  --validation-trials 256 \
  --trajectory-downsample 20 \
  --amortizer-name FINAL_DATASET_ONLY \
  --amortizer-iterations 1 \
  --steps-per-iteration 1 \
  --batch-size 1 \
  --save-frequency 1 \
  --device cuda \
  --skip-amortizer \
  --pair-seed "$PAIR_SEED"

dataset_dir="data/amortizer_cond/dataset/${RL_MODEL}_${RL_CHECKPOINT}_20_default_012345678_9"
compgen -G "$dataset_dir/tr_*mp_3sp_64ep_*.npz" >/dev/null || {
  printf 'Training dataset was not created: %s\n' "$dataset_dir" >&2
  exit 1
}
compgen -G "$dataset_dir/vd_100users_3_conds_256trials_*.npz" >/dev/null || {
  printf 'Validation dataset was not created: %s\n' "$dataset_dir" >&2
  exit 1
}

printf 'FINAL_DATASET_COMPLETE=%s\n' "$dataset_dir"
