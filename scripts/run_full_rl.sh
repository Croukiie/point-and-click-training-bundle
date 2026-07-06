#!/usr/bin/env bash
set -euo pipefail

NUM_CPU="${NUM_CPU:-8}"
RUN_NAME="${RUN_NAME:-pnc_mouse_full}"
AGENT_TYPE="${AGENT_TYPE:-classic}"
PNC_TYPE="${PNC_TYPE:-default}"
RL_SETTING="${RL_SETTING:-simple}"

python -m src.rl.sac_trainer \
  --num_cpu "$NUM_CPU" \
  --env_class_type PnCEnvDefault \
  --agent_type "$AGENT_TYPE" \
  --pnc_type "$PNC_TYPE" \
  --rl_setting "$RL_SETTING" \
  --model_name_prefix "$RUN_NAME"
