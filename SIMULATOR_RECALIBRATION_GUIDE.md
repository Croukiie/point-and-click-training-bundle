# Human-guided simulator recalibration

## Purpose

The human CPI recovery result should not be used to retrain the amortizer yet.
First, the simulator must be checked under the same controlled CPI contrast as
the human experiment.

This update performs that check in two stages:

1. It samples simulated participants and keeps each participant's nine non-CPI
   parameters fixed while running the same participant at 400, 800, and 1600
   CPI.
2. It ranks those shared profiles against the aggregate human pilot behavior.

The comparison includes absolute behavioral levels and within-profile CPI
trends. The current physical CPI mapping and the 24-inch Full HD mouse gain are
not changed.

## Compared features

- Movement time
- Target hit rate
- Click distance
- Maximum cursor speed
- Total absolute cursor turn
- Cursor path excess ratio

The last three features also receive a trend score across 400, 800, and 1600
CPI. This is important because the human pilot shows increasing cursor speed
and increasing trajectory curvature/path excess at higher CPI.

## Files added by this update

- `src/scripts/generate_fixed_cpi_calibration.py`: creates a matched fixed-CPI
  simulator dataset.
- `src/scripts/calibrate_simulator_to_humans.py`: scores profiles and produces
  plots and an exploratory restricted prior.
- `calibration/human_calibration_targets.csv`: aggregate pilot targets from the
  two current participants.
- `run_simulator_calibration_yonsei.slurm`: complete Yonsei job.
- `submit_simulator_calibration_yonsei.sh`: smoke/full submission helper.
- `watch_simulator_calibration_progress.sh`: percentage/progress monitor.

## Run on Yonsei

From the bundle directory, start with a smoke test:

```bash
cd ~/point-and-click-training-bundle
bash submit_simulator_calibration_yonsei.sh smoke
```

The submission stores the latest Slurm job ID automatically. Follow it with:

```bash
bash watch_simulator_calibration_progress.sh
```

The smoke test uses one simulated profile and two trials per CPI. Its score is
not scientifically meaningful. It only verifies that model loading, fixed CPI
simulation, plotting, and archiving work.

After `Calibration complete` appears, launch the full audit:

```bash
bash submit_simulator_calibration_yonsei.sh full
```

The default full audit uses 100 shared parameter profiles and 64 trials at each
CPI, for 19,200 simulated trials in total. It reuses the trained RL model
`pnc_rl_24fhd_cpi_26210DBCT` at its `best` checkpoint.

To use another RL run, set it before submission:

```bash
RL_MODEL=YOUR_RL_MODEL bash submit_simulator_calibration_yonsei.sh full
```

## Progress display

The log contains two percentage bars:

- `SIMULATION`: generation of the matched 400/800/1600 CPI profiles.
- `CALIBRATION`: scoring, exports, and figures.

The watcher can be stopped with `Ctrl+C`; this does not stop the Slurm job.

## Results

The job creates:

```text
data/calibration/simulator_human_JOBID/
├── dataset/
│   ├── fixed_cpi_profiles.npz
│   ├── fixed_cpi_profiles.yaml
│   └── fixed_cpi_profiles.manifest.json
└── results/
    ├── calibration_candidates.csv
    ├── calibration_condition_details.csv
    ├── calibrated_parameter_ranges.yaml
    ├── calibration_manifest.json
    ├── calibration_scores.png
    └── top_candidate_human_comparison.png
```

It also creates `simulator_calibration_JOBID.tar.gz` in the bundle root.

The main files are:

- `top_candidate_human_comparison.png`: human means and standard deviations
  against the best simulator profiles at each CPI.
- `calibration_candidates.csv`: one ranked row per shared parameter profile.
- `calibrated_parameter_ranges.yaml`: the 5th-95th percentile range of the
  best profiles. This is an exploratory pilot prior, not a final population
  estimate.
- `calibration_manifest.json`: counts, best score, and number of profiles that
  pass all hard checks.

## Decision after the audit

If several candidates pass the hard level checks and reproduce the positive
CPI trends, the next step is to regenerate the shared amortizer data using the
proposed restricted non-CPI ranges. A new 30-million-step RL training run is not
needed at this stage.

If no candidate reproduces the human CPI trends, changing the parameter prior
is insufficient. The next controlled test should retrain short RL variants with
reduced CPI policy conditioning, then repeat this exact calibration audit. The
mouse gain equation itself should remain unchanged because it already models
the physical count-to-cursor scaling.

Do not retrain the final amortizer until one of these simulator variants passes
the fixed-CPI audit.
