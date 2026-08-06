# Human CPI experiment guide

This protocol collects human point-and-click trajectories that can be compared
directly with the curvature amortizer.

## What is collected

- three counterbalanced CPI conditions: 400, 800, and 1600 CPI;
- 15 practice trials and 128 recorded trials per CPI;
- target radii restricted to 1.25--7 mm (about 9--51 pixels in diameter)
  to match the configured RL range and remove extreme target sizes;
- a confirmation of the physical CPI setting before every block;
- mouse model, declared polling rate, CPI-setting method, mousepad, Windows
  pointer speed, and pointer acceleration;
- lossless Windows Raw Input counts and reconstructed cursor positions;
- clicks, target geometry, timing, cursor speed, path length, and trial outcome;
- ten curvature features calculated from the full-frequency trajectory before
  the optional 20 Hz amortizer trajectory is created.

The 20 Hz export does not replace the raw data. Curvature is always calculated
from the full-resolution event stream.

## Before each participant

1. Use a 24-inch, 1920 x 1080 display at the configured 240 Hz refresh rate.
2. Create or verify mouse profiles for 400, 800, and 1600 CPI.
3. Keep the same mouse, mousepad, Windows pointer speed, and acceleration state
   for the whole session.
4. Close applications that can steal focus or add mouse processing.
5. Use the Windows Raw Input backend for real data. The browser fallback is for
   pilot testing only.

## First installation on Windows

Open PowerShell in the bundle directory:

```powershell
cd C:\Users\elisa\point-and-click-training-bundle
py -3.11 -m venv .venv-experiment
.\.venv-experiment\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-experiment.txt
```

## Launch the experiment

Activate the environment and launch the local recorder:

```powershell
cd C:\Users\elisa\point-and-click-training-bundle
.\.venv-experiment\Scripts\Activate.ps1
.\run_human_experiment_windows.ps1
```

The mousepad condition is left as `unspecified` for now. The page opens locally
in the browser. At every block, set the requested physical CPI and click the
button that explicitly confirms it.

If `.venv-experiment` is not present but the active local Python already has
the required packages, the launcher uses that Python installation directly.

The source data are saved under:

```text
data/human_experiment/<session_id>/
```

Do not edit these session folders manually.

Sessions collected with the restricted target interval use protocol version
`target_radius_matched_v2`. Keep the earlier pilot sessions as a separate
calibration dataset; do not merge the two protocol versions in one analysis.

Export only the restricted-radius sessions with:

```powershell
python -m src.experiment.export_human_amortizer_data --protocol-version target_radius_matched_v2
```

## Check and export the data

After one or more completed sessions:

```powershell
python -m src.experiment.quality_check_human_data
python -m src.experiment.export_human_amortizer_data
```

The second command writes a timestamped folder under
`data/human_experiment/amortizer_ready/` containing:

- `human_amortizer_stats.csv`: the 21 curvature-amortizer statistics;
- `human_amortizer_trajectories.csv`: movement-only trajectories resampled at
  20 Hz and grouped by `trial_uid`;
- `excluded_trials.csv`: rejected trials and the exact reason;
- `export_manifest.json`: counts, filters, CPI balance, and schema details.

The `cpi` column is kept only as ground truth for evaluating recovery. It must
not be included among the amortizer input features.

## Run the complete human/mouse inference report

After exporting the human data, run both the curvature-only and the
curvature-plus-trajectory amortizers with:

```powershell
python -m src.experiment.infer_human_mouse
```

The command automatically selects the newest `amortizer_ready` export, uses
checkpoint 200 from both models, evaluates 16, 32, 64, and 128 trials, and
writes the following results inside the export folder:

- CPI recovery CSVs and `human_cpi_inference.png`;
- all ten inferred simulator parameters in long and full-trial CSV formats;
- `human_mouse_model_parameters.png` for CPI, rotation gains, and plan correction;
- `human_behavioral_parameters.png` for motor/timing noise and reward/cost terms;
- `human_observed_mouse_metrics.csv` for the mouse model, declared polling rate,
  observed Windows Raw Input delivery rate, Windows settings, protocol mouse
  gain, speed, and curvature.

Only CPI is a physical sensor setting inferred by the amortizer. Rotation gains
and plan correction describe the simulator's mouse-mapping model. Motor/timing
noise and reward/cost terms describe the participant/model. The hardware fields
and trajectory metrics in `human_observed_mouse_metrics.csv` are measured or
declared metadata, not latent parameters inferred by the amortizer.

The Raw Input delivery rate is not a direct USB polling-rate measurement.
Windows, the device driver, or the Python callback may batch several 1000 Hz
reports into fewer `WM_INPUT` messages. Verify the physical polling rate with a
dedicated mouse polling tester when it is an experimental ground truth.

The older command `python -m src.experiment.infer_human_cpi` remains available
and produces the same complete report.

## Compare human participants

After exporting all completed sessions, create a compact participant comparison
with:

```powershell
python -m src.experiment.plot_human_participant_comparison
```

The command compares movement time, hit rate, click error, maximum cursor speed,
trajectory turning, and path excess at every CPI. Points show participant means
and error bars show the standard error across trials.

To add the trained RL simulator population to the same comparison, run:

```powershell
python -m src.experiment.plot_human_model_comparison
```

This uses the largest local RL validation dataset. For each experimental CPI,
it averages simulated users within ±100 CPI. Model uncertainty is calculated
across simulated users rather than treating their repeated trials as independent.
This is a population-level model check, not a personalized posterior prediction.

To create publication-style CPI recovery plots for the human pilot data, run:

```powershell
python -m src.experiment.plot_cpi_recovery_pilot
```

It creates separate summary-only and with-trajectory figures, plus a combined
figure. Participant lines use all available trials; black diamonds show the
mean ± standard deviation across participants at each physical CPI.

## Simulate the exact human targets

Run a quick posterior-predictive check on 32 matched trials per participant/CPI:

```powershell
python -m src.experiment.simulate_matched_human_tasks
```

The command uses the human cursor start, target position, target radius, and
reaction time for every selected trial. It fixes CPI to the known experimental
condition and uses the other fitted parameters from the trajectory amortizer.
Solid graph lines are human observations; dashed lines are the fitted RL model
on the identical targets. Use `--trials-per-block 128` for the full comparison.
