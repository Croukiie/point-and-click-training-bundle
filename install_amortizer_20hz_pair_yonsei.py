#!/usr/bin/env python3
"""Install the matched 20 Hz trajectory/summary amortizer launcher on Yonsei."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SIMULATOR = ROOT / "src" / "agent" / "pnc_simulator.py"
SLURM = ROOT / "run_shared_cpi_2step_yonsei.slurm"
SUBMIT = ROOT / "submit_shared_cpi_pair_yonsei.sh"


SUMMARY_PREFIX = '''    def get_summary_result(self, downsample=None):
        target_vel = self.initial_task_cond["target_spd"] * cos_sin_array(self.initial_task_cond["target_dir"])
        trajectory = self.get_trajectory_result(downsample=downsample)
        timestamp = trajectory[FIELD.TIMESTAMP].to_numpy(dtype=float)
        cursor_pos = trajectory[
            [FIELD.CURSOR_POS_X, FIELD.CURSOR_POS_Y]
        ].to_numpy(dtype=float)
        hand_pos = trajectory[
            [FIELD.HAND_POS_X, FIELD.HAND_POS_Y]
        ].to_numpy(dtype=float)

        cursor_segment_length = np.linalg.norm(np.diff(cursor_pos, axis=0), axis=1)
        hand_segment_length = np.linalg.norm(np.diff(hand_pos, axis=0), axis=1)
        traverse_length = float(np.sum(cursor_segment_length))
        curvature_features = cursor_curvature_features(
            cursor_pos,
            target_position=self.initial_task_cond["target_pos"],
        )
        hand_traverse_length = float(np.sum(hand_segment_length))
        click_rel_endpt = self.result.target_endpt - self.result.click_endpt
        target_distance = np.linalg.norm(self.initial_task_cond["target_pos"] - self.initial_task_cond["cursor_pos"])
        iod = np.log2(target_distance / (2 * self.initial_task_cond["target_rad"]) + 1)

        sampled_dt = np.diff(timestamp)
        valid_dt = sampled_dt > 0
        if np.any(valid_dt):
            max_cursor_spd = float(
                np.max(cursor_segment_length[valid_dt] / sampled_dt[valid_dt])
            )
            max_hand_spd = float(
                np.max(hand_segment_length[valid_dt] / sampled_dt[valid_dt])
            )
        else:
            max_cursor_spd = 0.0
            max_hand_spd = 0.0

'''


SUBMIT_SCRIPT = r'''#!/usr/bin/env bash
# Generate one matched 20 Hz dataset, then train trajectory and summary-only
# shared-CPI amortizers from exactly the same simulated trials.

set -euo pipefail

BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BUNDLE_DIR"

MODE="${1:-smoke}"
RL_MODEL="${RL_MODEL:-pnc_rl_24fhd_cpi_hidden_short}"
RL_CHECKPOINT="${RL_CHECKPOINT:-best}"
TRAJECTORY_HZ="${TRAJECTORY_HZ:-20}"
PAIR_TAG="${PAIR_TAG:-$(date +%y%m%d_%H%M%S)}"
AMORTIZER_BASE="${AMORTIZER_BASE:-PTE_SHARED_CPI_2STEP_20HZ_${PAIR_TAG}}"

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

checkpoint_path="data/pnc_model/$RL_MODEL/$RL_CHECKPOINT/${RL_CHECKPOINT}_model.zip"
if [[ ! -f "$checkpoint_path" ]]; then
  printf 'Missing RL checkpoint: %s\n' "$checkpoint_path" >&2
  printf 'Available best checkpoints:\n' >&2
  find data/pnc_model -path '*/best/best_model.zip' -print 2>/dev/null >&2 || true
  exit 1
fi

common="ALL,RL_MODEL=$RL_MODEL,RL_CHECKPOINT=$RL_CHECKPOINT"
common+=",DATA_CPU=${DATA_CPU:-64},TRAIN_MAIN_PARAMS=$TRAIN_MAIN_PARAMS"
common+=",TRIALS_PER_BLOCK=$TRIALS_PER_BLOCK"
common+=",VALIDATION_USERS=$VALIDATION_USERS,VALIDATION_TRIALS=$VALIDATION_TRIALS"
common+=",TRAJECTORY_HZ=$TRAJECTORY_HZ"
common+=",AMORTIZER_ITERATIONS=$AMORTIZER_ITERATIONS"
common+=",STEPS_PER_ITERATION=$STEPS_PER_ITERATION,BATCH_SIZE=$BATCH_SIZE"
common+=",SAVE_FREQUENCY=$SAVE_FREQUENCY,AMORTIZER_NAME=$AMORTIZER_BASE"

data_submission="$(
  sbatch --parsable \
    --job-name="pnc_20hz_data_${MODE}" \
    --export="$common,VARIANT=trajectory,SKIP_DATA=0,SKIP_AMORTIZER=1" \
    run_shared_cpi_2step_yonsei.slurm
)"
data_job="${data_submission%%;*}"

trajectory_submission="$(
  sbatch --parsable \
    --dependency="afterok:$data_job" \
    --job-name="pnc_20hz_traj_${MODE}" \
    --export="$common,VARIANT=trajectory,SKIP_DATA=1,SKIP_AMORTIZER=0" \
    run_shared_cpi_2step_yonsei.slurm
)"
trajectory_job="${trajectory_submission%%;*}"

summary_submission="$(
  sbatch --parsable \
    --dependency="afterok:$data_job" \
    --job-name="pnc_20hz_summary_${MODE}" \
    --export="$common,VARIANT=summary-only,SKIP_DATA=1,SKIP_AMORTIZER=0" \
    run_shared_cpi_2step_yonsei.slurm
)"
summary_job="${summary_submission%%;*}"

printf '%s %s %s\n' "$data_job" "$trajectory_job" "$summary_job" \
  > .last_shared_cpi_pair_jobs

printf 'Submitted matched 20 Hz shared-CPI pair (%s).\n' "$MODE"
printf 'RL model: %s (%s)\n' "$RL_MODEL" "$RL_CHECKPOINT"
printf 'Dataset job:      %s\n' "$data_job"
printf 'Trajectory job:   %s (after dataset)\n' "$trajectory_job"
printf 'Summary-only job: %s (after dataset)\n' "$summary_job"
printf 'Amortizer base:   %s\n' "$AMORTIZER_BASE"
printf 'Status: squeue -j %s,%s,%s\n' "$data_job" "$trajectory_job" "$summary_job"
printf 'Dataset progress: tail -F slurm-pnc-shared-cpi-%s.out\n' "$data_job"
printf 'Trajectory log:   tail -F slurm-pnc-shared-cpi-%s.out\n' "$trajectory_job"
printf 'Summary log:      tail -F slurm-pnc-shared-cpi-%s.out\n' "$summary_job"
'''


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Required file is missing: {path}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def build_simulator(text: str) -> str:
    new_signature = "    def get_summary_result(self, downsample=None):"
    old_signature = "    def get_summary_result(self):"
    action_marker = '        action_kc = self.actions["kc"].to_numpy().astype(int)'

    if new_signature in text:
        updated = text
    else:
        try:
            start = text.index(old_signature)
            end = text.index(action_marker, start)
        except ValueError as exc:
            raise SystemExit(f"Could not locate simulator summary block: {exc}") from exc
        updated = text[:start] + SUMMARY_PREFIX + text[end:]

    old_call = "summ.append(rec.get_summary_result())"
    new_call = "summ.append(rec.get_summary_result(downsample=downsample))"
    if new_call not in updated:
        updated = replace_once(updated, old_call, new_call, "simulation result")

    ast.parse(updated, filename=str(SIMULATOR))
    return updated


def build_slurm(text: str) -> str:
    updated = text
    if 'TRAJECTORY_HZ="${TRAJECTORY_HZ:-20}"' not in updated:
        anchor = 'VALIDATION_TRIALS="${VALIDATION_TRIALS:-256}"\n'
        updated = replace_once(
            updated,
            anchor,
            anchor + 'TRAJECTORY_HZ="${TRAJECTORY_HZ:-20}"\n',
            "validation-trials setting",
        )

    if 'SKIP_AMORTIZER="${SKIP_AMORTIZER:-0}"' not in updated:
        anchor = 'SKIP_DATA="${SKIP_DATA:-0}"\n'
        updated = replace_once(
            updated,
            anchor,
            anchor + 'SKIP_AMORTIZER="${SKIP_AMORTIZER:-0}"\n',
            "skip-data setting",
        )

    if '--trajectory-downsample "$TRAJECTORY_HZ"' not in updated:
        anchor = '  --validation-trials "$VALIDATION_TRIALS"\n'
        updated = replace_once(
            updated,
            anchor,
            anchor + '  --trajectory-downsample "$TRAJECTORY_HZ"\n',
            "validation-trials argument",
        )

    if 'args+=(--skip-amortizer)' not in updated:
        anchor = 'if [[ "$SKIP_DATA" == "1" ]]; then\n  args+=(--skip-data)\nfi\n'
        updated = replace_once(
            updated,
            anchor,
            anchor
            + 'if [[ "$SKIP_AMORTIZER" == "1" ]]; then\n'
            + '  args+=(--skip-amortizer)\n'
            + 'fi\n',
            "skip-data block",
        )
    return updated


def check_bash(contents: str, label: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write(contents)
        temporary = Path(handle.name)
    try:
        subprocess.run(["bash", "-n", str(temporary)], check=True)
    finally:
        temporary.unlink(missing_ok=True)


def backup(path: Path) -> Path:
    destination = path.with_name(path.name + ".before_20hz_installer")
    if not destination.exists():
        shutil.copy2(path, destination)
    return destination


def main() -> int:
    require_file(SIMULATOR)
    require_file(SLURM)

    simulator_text = SIMULATOR.read_text(encoding="utf-8")
    slurm_text = SLURM.read_text(encoding="utf-8")
    updated_simulator = build_simulator(simulator_text)
    updated_slurm = build_slurm(slurm_text)

    check_bash(updated_slurm, "Slurm script")
    check_bash(SUBMIT_SCRIPT, "submission script")

    simulator_backup = backup(SIMULATOR)
    slurm_backup = backup(SLURM)

    SIMULATOR.write_text(updated_simulator, encoding="utf-8", newline="\n")
    SLURM.write_text(updated_slurm, encoding="utf-8", newline="\n")
    SUBMIT.write_text(SUBMIT_SCRIPT, encoding="utf-8", newline="\n")
    os.chmod(SUBMIT, SUBMIT.stat().st_mode | 0o111)

    print("UPDATE_OK")
    print(f"Simulator backup: {simulator_backup}")
    print(f"Slurm backup:     {slurm_backup}")
    print("Trajectory input and trajectory-derived summaries: 20 Hz")
    print(f"Pair launcher:    {SUBMIT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Bash syntax validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
