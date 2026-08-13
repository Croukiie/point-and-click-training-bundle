# Controlled CPI ablation matrix

This is a selective port of the colleague repository's testing and ablation
organization. It does not merge the repositories and does not replace the RL,
simulator, inference configuration, human experiment, or trained models.

## Imported ideas

- Test geometric feature contracts independently of model training.
- Test shared-versus-block-specific parameter partitioning explicitly.
- Test duplicate timestamps, stationary trajectories, and terminal samples.
- Define an ablation as a matrix in which one input family changes at a time.
- Use one raw dataset, one seed, one training budget, and one preprocessing rate.
- Use Slurm arrays instead of maintaining one job script per feature subset.
- Separate smoke, diagnostic, and full budgets.

## Deliberately not imported

- The colleague `src/config/inference.yaml` and its `derived_stat` schema.
- Their `src/inference/trajectory_features.py` formulas.
- Their pilot conversion/fitting scripts and pilot-data directory layout.
- Their RL checkpoints, simulator code, datasets, and training configurations.
- Their legacy independent and three-step inference pipelines.

Those files use different APIs and copying them would silently change the
meaning of the current ten curvature features.

## Matched profiles

| Array index | Profile | Retained curvature features |
|---:|---|---:|
| 0 | `speedpath` | 0 |
| 1 | `core3` | 3 |
| 2 | `clean7` | 7 |
| 3 | `full10` | 10 |

All four profiles are summary-only at 20 Hz and use the same eleven base
statistics, shared nine-parameter participant vector, and block-specific CPI.

## Run tests

```bash
.venv/bin/python -m unittest discover -s tests \
  -p 'test_imported_ablation_contracts.py' -v
```

## Print the matrix

```bash
.venv/bin/python -m src.scripts.shared_cpi_ablation_matrix --print-matrix
```

## Submit on Yonsei

Always run smoke first:

```bash
bash submit_shared_cpi_ablation_matrix_yonsei.sh smoke
bash watch_shared_cpi_ablation_matrix.sh
```

After all four smoke tasks pass, launch the complete matrix explicitly:

```bash
bash submit_shared_cpi_ablation_matrix_yonsei.sh full
```

The existing simulation dataset is reused intentionally for a controlled
feature ablation. For the final corrected experiment, first retrain the hidden
CPI RL and regenerate this dataset after installing the target-radius fix.
