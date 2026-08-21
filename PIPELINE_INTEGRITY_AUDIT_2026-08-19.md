# Point-and-Click CPI Pipeline Integrity Audit

Date: 19 August 2026  
Scope: the active PnC RL, 20 Hz shared-CPI amortizer, simulator replay, and local human experiment/export pipeline.

## Executive summary

The audit found several issues that could make simulated CPI recovery look better than human CPI recovery, reduce reproducibility, or make a human trial differ from its nominal simulated task. The important issues have been corrected in the local source tree.

The corrections change the RL reward/observation contract, the simulated dataset contract, and the human task protocol. Therefore, the scientifically clean comparison requires a new CPI-visible RL, a newly generated shared dataset, new matched speed/path and CORE3 amortizers, and new human trials collected with protocol `simulator_geometry_exact_v3`.

The currently queued Yonsei visible-CPI chain was submitted before this full audit. Its outputs should be retained as a pre-integrity baseline, not renamed as the corrected v3 result.

## Corrections implemented

### 1. Target sampling and task geometry

- Fitts ID is now sampled once and held fixed while invalid cursor/target geometries are rejected. Previously, rejection also resampled ID, which biased the realized ID distribution.
- Both simulation and the new human protocol enforce the same target-radius range: 1.25 to 7 mm.
- The human experiment no longer recomputes and silently clamps target radius from the cursor position left by the previous trial.
- Each logical human trial now uses a sampled start target paired with the sampled task target. After a successful start click, the virtual cursor is explicitly snapped to the sampled start-target center. This preserves the exact simulated distance, radius, and Fitts ID.
- The new human protocol is versioned as `simulator_geometry_exact_v3`, so it cannot be confused with earlier data.

### 2. Exact replay of mouse parameters

Simulator replay now receives the exact CPI, `mouse_rotation_gain_x`, `mouse_rotation_gain_y`, and `mouse_plan_correction` used in the original RL rollout. Curvature and path summaries are therefore computed from the same physical mouse mapping that produced the episode.

### 3. Shared trajectory preprocessing

Simulation, human export, and inference now use one endpoint-preserving preprocessing contract:

1. identify the movement interval;
2. remove the reaction interval before resampling;
3. resample on an exact `1 / 20 s` grid;
4. preserve the true movement-start position;
5. append the exact click timestamp and click position as the terminal sample;
6. recompute path length, maximum cursor speed, and curvature features from that exported 20 Hz trajectory.

The previous `linspace` implementation produced approximately 20 Hz samples whose interval depended on trial duration. It has been removed from the PnC inferers.

### 4. Trial termination and motor effort

- Native Raw Input processing now stops at the first click rather than processing later movement events already present in the same input batch.
- A terminal click is charged only for the executed part of the motor plan. Previously, a click inside a 100 ms decision interval could receive the effort cost of the entire interval.
- Simulated timeouts are retried up to 50 times. Dataset generation now fails explicitly if a truncated episode still remains, because the human amortizer export contains completed clicks rather than timeout trials.

### 5. RL training and evaluation

- The default PnC agent/environment names now refer to valid configurations (`classic` and `PnCEnvDefault`).
- RL training has an explicit seed, and the seed is saved in `reproducibility.yaml`.
- Evaluation now runs the configured number of episodes. The previous code divided this count by the number of training environments even though the evaluation environment count was one.
- Observations are clipped to the declared Gymnasium range `[-1, 1]`.
- Environment reset now applies the supplied seed to the legacy NumPy samplers.

### 6. Amortizer training and datasets

- Gradient clipping now uses the configured finite threshold. The previous trainer overwrote the configuration with infinity, effectively disabling clipping.
- Dataset worker seeds are deterministic and their SHA-256 digest is stored in dataset metadata.
- If multiple matching dataset files exist, the loader reports this and selects the newest file instead of silently concatenating independent reruns.
- Episode sampling uses replacement only when more samples are requested than are available.
- The new dataset path includes the contract name `movement_exact20_endpoint_clickonly_v3`. Old and new preprocessing outputs cannot be silently mixed.
- RL checkpoints are loaded lazily and on CPU for simulation. Parameter inference no longer needs to deserialize the RL policy merely to recover parameter ranges, which avoids the NumPy checkpoint error encountered between the local and Yonsei environments.

### 7. Recovery metrics

The former recovery score was squared Pearson correlation. That score is high for a reversed relationship and does not measure calibration. Recovery figures and logs now report:

- predictive R-squared, `1 - SSE / SST`;
- signed Pearson correlation;
- regression slope and intercept;
- MAE;
- RMSE;
- CPI exact-order and pairwise-order accuracy for the shared model.

This prevents a decreasing inferred CPI trend from being described as good recovery merely because its unsigned correlation is large.

### 8. Completed-click replacement sampling

- A failed dataset job exposed that the previous timeout retry replayed the exact same task, user state, and deterministic policy up to 50 times. Such a replay cannot turn a deterministic timeout into a click.
- Participant parameters, shared parameters, and block-specific CPI now remain fixed while each truncated episode is replaced by a freshly sampled task and user state.
- Truncated episodes are still excluded rather than silently entering the click-only dataset. Generation fails explicitly if the requested number of completed-click trials cannot be collected within the configured attempt budget.
- Dataset metadata now records the completion sampling policy, total requested trials, total attempts, and acceptance rate. This makes any completion-conditioned simulation selection visible during later simulation-to-human audits.

## Normalization

The min-max normalization formula itself was correct and was not replaced. Human values outside the configured simulated ranges are deliberately not clipped at the feature-normalization stage, because clipping would hide the simulation-to-human domain gap. These out-of-range rates should continue to be reported by the feature audit.

## Validation performed

- Python source compilation completed successfully.
- The complete local unit-test suite passed.
- Tests cover exact 20 Hz timing, click-endpoint preservation, duplicate timestamps, fixed-ID/radius sampling, exact start/task geometry, visible versus hidden CPI observations, and the reversed-recovery metric case.
- Native environment smoke tests completed for both `PnCEnvDefault` and `PnCEnvCpiHidden`, with observations inside the declared space and finite rewards.
- All new Yonsei shell scripts pass syntax checks.
- Both speed/path-only and CORE3 runners pass dry-run design checks.

## Interpretation of previous results

Previous results remain useful as diagnostics, but they should not be pooled with the corrected v3 results. In particular:

- older human trials may contain dynamically clamped target radii and do not use the exact paired start geometry;
- older simulated summaries may use different reaction cropping, endpoint sampling, terminal effort, or replay parameters;
- older recovery plots may show squared correlation rather than predictive recovery;
- older datasets do not carry the v3 contract identifier.

These differences can contribute to the reversed or flat human CPI ordering, but they do not prove that preprocessing was the only cause. A genuine simulation-to-human domain gap may remain after correction.

## Recommended next experiment

1. Keep the current Yonsei chain as the pre-integrity baseline.
2. Install the integrity update only after that chain has left the queue and finished.
3. Submit the new one-command CPI-visible chain.
4. Train one 30M-step `PnCEnvDefault` RL using the corrected sampler and terminal dynamics.
5. Generate one versioned shared dataset with a fixed seed.
6. Train matched speed/path-only and CORE3 amortizers from that same dataset.
7. Collect new human data with `simulator_geometry_exact_v3` at 400, 800, and 1600 CPI.
8. Compare simulated recovery, human CPI ordering, feature slopes, and out-of-range rates using the corrected signed/calibration-aware metrics.

## Scope limitation

This audit covers the active PnC pipeline. Historical `ans_*` analysis code and unused legacy experimental branches were not treated as part of the current CPI experiment and should not be used for the final comparison without a separate audit.
