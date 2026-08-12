V12 SUMMARY-ONLY: AUTOMATIC SMOKE -> FULL CHAIN

Install this small update in the bundle, then run:

  chmod +x submit_v12_summary_hidden_yonsei.sh
  chmod +x submit_v12_smoke_then_full_yonsei.sh
  bash -n submit_v12_summary_hidden_yonsei.sh
  bash -n submit_v12_smoke_then_full_yonsei.sh

  RUN_BASE=v12_summary_hidden bash submit_v12_smoke_then_full_yonsei.sh

Both jobs are submitted immediately. The full job has an afterok dependency and
therefore starts only if the smoke test succeeds. It generates its own full-size
dataset with REUSE_DATA=0.

Inspect both jobs:

  read SMOKE_JOB FULL_JOB < .last_v12_smoke_full_jobs
  squeue -j "$SMOKE_JOB,$FULL_JOB" -o "%.18i %.10T %.12M %.40R"

Smoke log:

  tail -F "slurm-pnc-shared-cpi-${SMOKE_JOB}.out"

Once the smoke job is complete, full log:

  tail -F "slurm-pnc-shared-cpi-${FULL_JOB}.out"

If the smoke job fails, the full job will remain DependencyNeverSatisfied and
will not train. It can then be cancelled with:

  scancel "$FULL_JOB"
