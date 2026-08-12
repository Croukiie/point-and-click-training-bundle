# CLEAN7 curvature ablation

This variant keeps the existing RL policy and raw 20 Hz simulated dataset. It
trains a new summary-only shared-CPI amortizer after removing three problematic
curvature features:

- `cursor_p90_abs_turn_rad`: strongest human/simulation distribution mismatch;
- `cursor_path_excess_ratio`: 11× human/simulation mean difference and many
  exact-zero simulated values;
- `cursor_total_abs_turn_rad`: perfectly redundant with
  `cursor_mean_abs_turn_rad` in the fixed-size 20 Hz representation.

The seven retained curvature features are net turn, mean absolute turn, turn
energy, maximum deviation, early turn, middle turn, and late turn. All ordinary
task summaries are retained. Existing models continue to use the `full` feature
profile and are not modified.

## Install on Yonsei

From the bundle directory:

```bash
cp src/scripts/shared_cpi_2step_run.py \
  src/scripts/shared_cpi_2step_run.py.before_clean7
cp run_shared_cpi_2step_yonsei.slurm \
  run_shared_cpi_2step_yonsei.slurm.before_clean7

tar -xzf curvature_clean7_ablation_update.tar.gz
chmod +x submit_shared_cpi_clean7_yonsei.sh

.venv/bin/python -m compileall -q src/scripts/shared_cpi_2step_run.py
bash -n run_shared_cpi_2step_yonsei.slurm
bash -n submit_shared_cpi_clean7_yonsei.sh
```

## Smoke test

The previous raw dataset is reused, so the RL policy and simulation dataset are
not regenerated.

```bash
RL_MODEL=pnc_rl_24fhd_cpi_26210DBCT \
REUSE_DATA=1 \
bash submit_shared_cpi_clean7_yonsei.sh smoke
```

Monitor it with:

```bash
JOBID=$(cat .last_shared_cpi_clean7_job)
squeue -j "$JOBID"
tail -F "slurm-pnc-shared-cpi-${JOBID}.out"
```

The log must display `Feature profile: clean7` and the three removed features.

## Full amortizer training

Run this only after the smoke test ends with `SHARED_CPI_COMPLETE=`:

```bash
RL_MODEL=pnc_rl_24fhd_cpi_26210DBCT \
REUSE_DATA=1 \
RUN_TAG=visible_cpi_clean7_full \
bash submit_shared_cpi_clean7_yonsei.sh full
```

The new model is independent of the existing full-feature amortizer. The RL
policy does not need to be retrained.
