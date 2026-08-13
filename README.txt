YONSEI CHAIN: HIDDEN-CPI RL -> CORE3 AMORTIZER

One submission queues three dependent jobs:
  1. A 30-million-step RL trained with PnCEnvCpiHidden on Yonsei.
  2. A two-iteration CORE3 smoke amortizer.
  3. The 200-iteration CORE3 full amortizer.

The amortizer is summary-only at 20 Hz. It keeps only these curvature cues:
  cursor_turn_energy_rad2
  cursor_early_mean_abs_turn_rad
  cursor_late_mean_abs_turn_rad

True CPI is never an amortizer input. The nine non-CPI participant parameters
are shared across three anonymous blocks, with one inferred CPI per block.

Launch:
  RUN_TAG=hidden_rl_core3 bash submit_hidden_rl_then_core3_yonsei.sh

Inspect:
  read RL_MODEL RL_JOB SMOKE_JOB FULL_JOB < .last_hidden_rl_core3_chain
  squeue -j "$RL_JOB,$SMOKE_JOB,$FULL_JOB" -o "%.18i %.10T %.12M %.40R"
  tail -F "slurm-pnc-bundle-${RL_JOB}.out"
