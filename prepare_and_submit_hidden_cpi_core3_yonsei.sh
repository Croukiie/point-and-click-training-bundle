#!/usr/bin/env bash
# Convert the Windows/NumPy-2 RL checkpoint to Yonsei's NumPy runtime, verify
# it in a clean process, and submit the CORE3 smoke -> full chain.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

SOURCE_MODEL="${SOURCE_MODEL:-pnc_rl_24fhd_cpi_hidden_local_30m_26222DSFT}"
COMPAT_MODEL="${COMPAT_MODEL:-${SOURCE_MODEL}_yonsei_compat}"
SOURCE_DIR="data/pnc_model/$SOURCE_MODEL"
COMPAT_DIR="data/pnc_model/$COMPAT_MODEL"
SOURCE_CHECKPOINT="$SOURCE_DIR/best/best_model.zip"
COMPAT_CHECKPOINT="$COMPAT_DIR/best/best_model.zip"

for required in \
  "$SOURCE_CHECKPOINT" \
  "$SOURCE_DIR/env_info.yaml" \
  "$SOURCE_DIR/train_config.yaml" \
  submit_hidden_cpi_core3_yonsei.sh
do
  if [[ ! -f "$required" ]]; then
    printf 'Missing required file: %s\n' "$required" >&2
    exit 1
  fi
done

mkdir -p "$COMPAT_DIR/best"
cp "$SOURCE_DIR/env_info.yaml" "$COMPAT_DIR/env_info.yaml"
cp "$SOURCE_DIR/train_config.yaml" "$COMPAT_DIR/train_config.yaml"
sed -i 's/\r$//' "$COMPAT_DIR/env_info.yaml" "$COMPAT_DIR/train_config.yaml"
if [[ -f "$SOURCE_DIR/best/best_eval_summary.yaml" ]]; then
  cp "$SOURCE_DIR/best/best_eval_summary.yaml" "$COMPAT_DIR/best/"
fi

printf '%s\n' '[1/3] Converting the NumPy-2 checkpoint for Yonsei'
SOURCE_CHECKPOINT="$SOURCE_CHECKPOINT" \
COMPAT_CHECKPOINT="$COMPAT_CHECKPOINT" \
.venv/bin/python - <<'PY'
import importlib
import os
import sys

import numpy
from stable_baselines3 import SAC

# Install these aliases only after pandas/SB3 have initialized normally.
core = importlib.import_module("numpy.core")
sys.modules.setdefault("numpy._core", core)
setattr(numpy, "_core", core)
for name in ("multiarray", "numeric"):
    module = importlib.import_module(f"numpy.core.{name}")
    sys.modules.setdefault(f"numpy._core.{name}", module)

random_pickle = importlib.import_module("numpy.random._pickle")
original_bit_generator_ctor = random_pickle.__bit_generator_ctor

def compatible_bit_generator_ctor(bit_generator="MT19937"):
    if isinstance(bit_generator, type):
        return bit_generator()
    return original_bit_generator_ctor(bit_generator)

random_pickle.__bit_generator_ctor = compatible_bit_generator_ctor

source = os.environ["SOURCE_CHECKPOINT"]
destination = os.environ["COMPAT_CHECKPOINT"]
model = SAC.load(source, device="cpu")
model.save(destination)
print(f"Converted checkpoint: {destination}")
PY

printf '%s\n' '[2/3] Verifying the converted checkpoint without aliases'
COMPAT_CHECKPOINT="$COMPAT_CHECKPOINT" .venv/bin/python - <<'PY'
import os
from stable_baselines3 import SAC

checkpoint = os.environ["COMPAT_CHECKPOINT"]
model = SAC.load(checkpoint, device="cpu")
print(f"CHECKPOINT_COMPATIBLE={checkpoint}")
print(f"POLICY_CLASS={type(model.policy).__name__}")
PY

env_class="$(awk -F': *' '/^env_class:/{print $2; exit}' "$COMPAT_DIR/env_info.yaml" | tr -d '\r')"
if [[ "$env_class" != "PnCEnvCpiHidden" ]]; then
  printf 'Expected PnCEnvCpiHidden, found: <%s>\n' "${env_class:-missing}" >&2
  exit 1
fi

# Remove only a previous full job whose dependency has become impossible.
if [[ -f .last_hidden_cpi_core3_jobs ]]; then
  read -r old_smoke old_full < .last_hidden_cpi_core3_jobs || true
  if [[ -n "${old_full:-}" ]]; then
    old_reason="$(squeue -h -j "$old_full" -o '%r' 2>/dev/null | head -1)"
    if [[ "$old_reason" == DependencyNeverSatisfied ]]; then
      scancel "$old_full"
      printf 'Cancelled obsolete blocked job: %s\n' "$old_full"
    fi
  fi
fi

printf '%s\n' '[3/3] Submitting smoke -> full'
RL_MODEL="$COMPAT_MODEL" \
RUN_TAG="hidden_core3_compat_$(date +%y%m%d_%H%M%S)" \
bash submit_hidden_cpi_core3_yonsei.sh
