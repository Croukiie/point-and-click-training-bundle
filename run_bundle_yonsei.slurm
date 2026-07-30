#!/usr/bin/env bash
#SBATCH --job-name=pnc_bundle
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-pnc-bundle-%j.out
#SBATCH --error=slurm-pnc-bundle-%j.err

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$BUNDLE_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

NUM_CPU="${NUM_CPU:-${SLURM_CPUS_PER_TASK:-64}}"
RUN_NAME="${RUN_NAME:-pnc_bundle_yonsei}"
AGENT_TYPE="${AGENT_TYPE:-classic}"
PNC_TYPE="${PNC_TYPE:-default}"
RL_SETTING="${RL_SETTING:-default}"
STEPS="${STEPS:-30000000}"
DEVICE="${DEVICE:-cuda}"

"$PYTHON_BIN" - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this Slurm allocation.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

command=(
  "$PYTHON_BIN" -m src.rl.sac_trainer
  --num_cpu "$NUM_CPU"
  --env_class_type PnCEnvDefault
  --agent_type "$AGENT_TYPE"
  --pnc_type "$PNC_TYPE"
  --rl_setting "$RL_SETTING"
  --model_name_prefix "$RUN_NAME"
  --steps "$STEPS"
  --device "$DEVICE"
)

if [[ -n "${LOAD_MODEL:-}" ]]; then
  command+=(--load_model "$LOAD_MODEL")
fi
if [[ -n "${LOAD_REPLAY_BUFFER:-}" ]]; then
  command+=(--load_replay_buffer "$LOAD_REPLAY_BUFFER")
fi
if [[ "${RESET_NUM_TIMESTEPS:-0}" == "1" ]]; then
  command+=(--reset_num_timesteps)
fi
if [[ "${NO_TIME_TAG:-0}" == "1" ]]; then
  command+=(--no_time_tag)
fi

printf 'Launching:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
