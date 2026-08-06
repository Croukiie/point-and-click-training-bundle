# Shared-parameter CPI inference (2-step)

This pipeline tests whether CPI is confounded with the other latent mouse and
participant parameters in the independent amortizer.

## Inference structure

For each simulated or human participant, three anonymous trial blocks are
processed jointly.

Shared across all three blocks:

- `theta_m`
- `theta_c`
- `rew_succ`
- `rew_fail`
- `time_penalty`
- `effort_cost`
- `mouse_rotation_gain_x`
- `mouse_rotation_gain_y`
- `mouse_plan_correction`

Block-specific:

- `cpi`

The block label is only an anonymous grouping index. True CPI is not included
in the behavioral feature vector or the grouping label. Values such as 400,
800, and 1600 are used only after inference for recovery evaluation.

## Yonsei smoke test

From the bundle root on the login node:

```bash
bash submit_shared_cpi_2step_yonsei.sh smoke trajectory
```

The command prints the Slurm job ID. Follow its percentage progress with:

```bash
bash watch_shared_cpi_progress.sh JOB_ID
```

The smoke run validates the complete simulation, training, checkpoint, and
simulated-recovery pipeline. Its parameter estimates are not scientific
results.

## Full training

Only start this after the smoke job ends successfully:

```bash
bash submit_shared_cpi_2step_yonsei.sh full trajectory
```

The full configuration generates 43,690 simulated participants, with three CPI
blocks and 64 trials per block. This gives 8,388,480 simulated training trials,
approximately the same total number as the independent amortizer dataset.

The final model is written under:

```text
data/amortizer_cond/models/PTE_SHARED_CPI_2STEP_24FHD_TRAJ_*/
```

The completion line in the Slurm output is:

```text
SHARED_CPI_COMPLETE=<model name>
```

## Optional summary/curvature-only control

The same simulated dataset can be reused without the 20 Hz trajectory encoder:

```bash
SKIP_DATA=1 bash submit_shared_cpi_2step_yonsei.sh full summary-only
```

Do this only after the trajectory job has finished generating the shared
dataset. The curvature summary features remain unchanged.

## Simulated recovery

At each saved validation iteration, the trainer writes ordinary parameter R2
plots and CPI-ordering tables under the model's `results` directory. Files named
`trial*_cpi_ordering.csv` report:

- exact recovery of the three-block CPI order;
- pairwise CPI-order accuracy;
- participant-level Spearman ordering.

## Human shared-versus-independent comparison

After copying the completed shared model back into `data/amortizer_cond/models`,
run:

```bash
python -m src.experiment.infer_human_shared_cpi \
  --shared-model MODEL_NAME \
  --export-dir data/human_experiment/amortizer_ready/EXPORT_NAME \
  --trial-counts all
```

The command automatically selects the matching independent trajectory or
summary model. It records a hash for every selected trial subset, demonstrating
that both structures received exactly the same human trials. It produces a CPI
comparison graph, raw predictions, ordering diagnostics, and a JSON manifest.
