# Point-and-Click CPI Training Bundle: Modification Summary

## 1. Project objective

The original point-and-click reinforcement-learning bundle was adapted to study
how mouse CPI affects pointing behavior and whether CPI can be recovered from
observable cursor trajectories.

The main objectives were to:

1. make mouse motion physically consistent with real CPI-based sensors;
2. calibrate the environment for a 24-inch Full HD display;
3. train a CPI-aware reinforcement-learning policy;
4. create an amortized inference model capable of estimating CPI;
5. use cursor-path curvature as the main CPI-sensitive trajectory signal;
6. avoid leaking the true CPI into the amortizer inputs;
7. support human experiments using the same display and mouse calibration;
8. run the complete pipeline on the Yonsei AI Data Center.

## 2. Original state

Before the modifications, the bundle had the following characteristics:

- The base mouse gain was fixed at `4.16`.
- Mouse displacement was modeled as continuous physical displacement.
- Physical sensor movement was not quantized into integer mouse counts.
- Fractional, sub-count motion was not preserved between movements.
- The default initial mouse orientation was sampled between approximately
  `-30°` and `+20°`.
- Cursor-velocity observation ranges were limited to approximately
  `[-5, 5] m/s`.
- The inference pipeline used trajectories downsampled to 20 Hz.
- The default trajectory encoder received both cursor and physical hand
  coordinates.
- No robust cursor-curvature descriptors were available.
- No automatic CPI-curvature correlation analysis was provided.
- Human trial summaries did not contain curvature features.
- Mousepad surface condition was not recorded.
- No dedicated Yonsei Slurm launch scripts were included.

An intermediate curvature V2 implementation was removed. It was replaced by a
generic, reusable trajectory-feature implementation with explicit protections
against CPI leakage.

## 3. CPI and mouse-gain model

### 3.1 CPI definition

CPI means counts per inch. It is the number of integer sensor counts generated
when the mouse moves by one physical inch.

The physical distance represented by one count is:

```text
physical_distance_per_count = 0.0254 / CPI
```

Examples:

| CPI | Physical distance per count |
|---:|---:|
| 400 | 0.0635 mm |
| 800 | 0.03175 mm |
| 1600 | 0.015875 mm |

In the RL environment, CPI is treated as a user/model parameter sampled from
400 to 1600 in increments of 10. The primary human experiment conditions are
400, 800, and 1600 CPI.

### 3.2 24-inch Full HD calibration

The original base gain was:

```text
mouse_gain = 4.16
```

This historical value was replaced by a value derived from the physical and
pixel dimensions of a 24-inch, 16:9, 1920 x 1080 display:

```text
width  = 0.5313124474 m
height = 0.2988632517 m
```

The new gain is:

```text
mouse_gain =
    width_m * reference_CPI / (0.0254 * width_pixels)

mouse_gain =
    0.5313124474 * 400 / (0.0254 * 1920)
    = 4.357877685367455
```

The final reference configuration is:

```text
mouse_gain = 4.357877685367455
mouse_reference_cpi = 400
```

This calibration makes one raw count correspond to one horizontal Full HD
screen pixel at the reference configuration.

### 3.3 Effective cursor gain

The base gain remains constant, but the effective cursor-to-hand gain depends
on CPI:

```text
effective_gain = mouse_gain * CPI / reference_CPI
```

| CPI | Effective cursor/hand gain |
|---:|---:|
| 400 | 4.3579 |
| 800 | 8.7158 |
| 1600 | 17.4315 |

The same physical hand displacement therefore moves the cursor twice as far at
800 CPI and four times as far at 1600 CPI compared with 400 CPI.

## 4. Integer sensor-count simulation

### 4.1 Original behavior

Originally, continuous hand displacement was directly multiplied by the
effective gain:

```text
cursor_delta ~= effective_gain * hand_delta
```

This allowed arbitrarily small cursor movements and did not reproduce the
integer output of a physical mouse sensor.

### 4.2 Modified behavior

Physical sensor displacement is now converted into raw counts:

```text
raw_counts = sensor_displacement * CPI / 0.0254 + previous_residual
integer_counts = round(raw_counts)
next_residual = raw_counts - integer_counts
```

Cursor displacement is calculated from the integer counts:

```text
cursor_delta =
    integer_counts * mouse_gain * 0.0254 / reference_CPI
```

### 4.3 Sub-count residual

A persistent `mouse_count_residual` state was added.

For example, successive movements smaller than one count are accumulated
instead of being permanently lost. Only executed movements update this state.
Virtual rollouts used for planning work on a copy and do not alter the real
sensor residual.

The implementation supports both scalar segment-by-segment mapping and
vectorized trajectory mapping. Tests verify that the two implementations
produce equivalent results.

## 5. Initial mouse orientation

### 5.1 Modified default

The original asymmetric range:

```text
-30° to +20°
```

was replaced by:

```text
-15° to +15°
```

This provides a symmetric reference condition, reduces excessive orientation
variability, and makes the CPI effect easier to identify.

### 5.2 Added control configurations

`default_ori0` uses:

```text
fixed orientation = 0°
```

It provides a clean control condition without initial-orientation variability.

`default_ori_asym` uses:

```text
uniform orientation = -10° to +20°
```

It provides an intentionally asymmetric comparison condition.

## 6. Observation and normalization ranges

The original cursor-velocity observation range:

```text
[-5, +5] m/s
```

was expanded to:

```text
[-20, +20] m/s
```

The maximum cursor-speed normalization range used for inference was expanded
similarly.

At high CPI, cursor speed can be approximately four times the 400-CPI value.
The larger range prevents legitimate high-CPI observations from being clipped.

## 7. Trajectory-generation pipeline

The final simulated trajectory is generated as follows:

1. The RL agent creates a desired cursor movement.
2. The motor model converts it into a physical hand plan.
3. Parallel and perpendicular motor noise are added.
4. The initial motor plan is represented at 50 ms intervals.
5. At the end of the episode, it is interpolated to 1 ms resolution.
6. Each hand segment is transformed into the mouse-sensor coordinate system.
7. Mouse orientation and orientation changes are applied.
8. Physical sensor displacement is converted to integer counts using CPI.
9. The CPI-dependent gain converts counts into cursor displacement.
10. Cursor positions are accumulated while respecting screen boundaries.
11. A full-resolution replay is produced.

The replay contains:

- timestamps;
- cursor X/Y position and velocity;
- hand X/Y position and velocity;
- target X/Y position.

### Auxiliary amortizer trajectory

The auxiliary trajectory remains sampled at 20 Hz. It is linearly interpolated
from the 1-ms replay.

The final curvature profile exposes only:

- `timediff`;
- `cursor_pos_x`;
- `cursor_pos_y`;
- `target_rel_pos_x`;
- `target_rel_pos_y`.

Physical hand positions are excluded from this profile. The reaction-time
portion is removed before the trajectory is passed to the encoder.

## 8. Curvature-feature extraction

### 8.1 Motivation

Cursor-path curvature was identified as a CPI-sensitive behavioral signal.
Raw trajectories alone are difficult for the amortizer to exploit because
their lengths vary, local count noise produces oscillations, and physical hand
coordinates can indirectly reveal CPI.

A cursor-only curvature extractor was therefore implemented.

### 8.2 Input restrictions

The extractor accepts only:

- observable cursor X/Y positions;
- an optional observable target position.

It does not accept:

- CPI;
- mouse gain;
- physical hand positions;
- CPI-derived physical distance;
- target-parameter labels.

This is an explicit anti-leakage design decision.

### 8.3 Preprocessing

Before feature calculation:

1. non-finite samples are removed;
2. consecutive duplicate positions are removed;
3. the trajectory is resampled into 64 equal arc-length points;
4. a binomial smoothing kernel `[1, 4, 6, 4, 1] / 16` is applied;
5. the original endpoints are preserved;
6. consecutive segments are calculated;
7. signed turning angles are calculated with:

```text
turn_angle = atan2(cross(segment_i, segment_i+1),
                   dot(segment_i, segment_i+1))
```

Arc-length resampling reduces sensitivity to polling frequency. The principal
features are invariant to translation, rotation, and uniform spatial scale.

### 8.4 Added curvature features

| Feature | Calculation | Purpose |
|---|---|---|
| `cursor_total_abs_turn_rad` | Sum of absolute turning angles | Measures total path bending and correction |
| `cursor_net_turn_rad` | Sum of signed turning angles | Measures overall clockwise or counter-clockwise rotation |
| `cursor_mean_abs_turn_rad` | Mean absolute turning angle | Measures typical local curvature |
| `cursor_p90_abs_turn_rad` | 90th percentile absolute angle | Captures strong corrections without relying on one maximum |
| `cursor_turn_energy_rad2` | Mean squared turning angle | Gives greater weight to sharp direction changes |
| `cursor_max_deviation_normalized` | Maximum perpendicular deviation divided by reference distance | Measures deviation from the direct start-to-target route |
| `cursor_path_excess_ratio` | `max(0, path_length / reference_distance - 1)` | Measures extra distance caused by curvature and correction |
| `cursor_early_mean_abs_turn_rad` | Mean absolute turn in the first third | Captures initial planning corrections |
| `cursor_middle_mean_abs_turn_rad` | Mean absolute turn in the middle third | Captures online transport corrections |
| `cursor_late_mean_abs_turn_rad` | Mean absolute turn in the final third | Captures endpoint correction and click preparation |

## 9. Full-resolution curvature versus 20-Hz trajectory

Curvature is calculated from the complete 1-ms cursor replay before the
auxiliary trajectory is reduced to 20 Hz.

The full replay is used as the source and is geometrically resampled to 64
equal arc-length points for stable feature extraction.

Therefore:

- curvature does not depend on the 20-Hz encoder input;
- increasing the auxiliary trajectory to 100 Hz would not improve the
  resolution of the engineered curvature descriptors;
- 100 Hz would mainly increase storage, padding, memory use, and trajectory
  encoder computation.

## 10. New `curvature` inference profile

A new normalization and feature configuration named `curvature` was added.

### 10.1 Static per-trial features

The profile contains 21 static features.

Existing observable behavioral features:

1. `movement_time`
2. `target_hit`
3. `click_distance`
4. `target_rel_pos_x0`
5. `target_rel_pos_y0`
6. `click_rel_endpoint_x`
7. `click_rel_endpoint_y`
8. `target_radius`
9. `target_distance`
10. `traverse_length`
11. `max_cursor_speed`

Added curvature features:

12. `cursor_total_abs_turn_rad`
13. `cursor_net_turn_rad`
14. `cursor_mean_abs_turn_rad`
15. `cursor_p90_abs_turn_rad`
16. `cursor_turn_energy_rad2`
17. `cursor_max_deviation_normalized`
18. `cursor_path_excess_ratio`
19. `cursor_early_mean_abs_turn_rad`
20. `cursor_middle_mean_abs_turn_rad`
21. `cursor_late_mean_abs_turn_rad`

### 10.2 Auxiliary sequential features

1. `timediff`
2. `cursor_pos_x`
3. `cursor_pos_y`
4. `target_rel_pos_x`
5. `target_rel_pos_y`

### 10.3 Inputs excluded from the curvature profile

The following fields were intentionally excluded:

- `hand_pos_x`;
- `hand_pos_y`;
- `raw_path_length_m`;
- `raw_max_speed_m_s`;
- mouse gain;
- CPI.

Converting raw counts into physical hand metres requires CPI. Supplying that
conversion to a model whose task is to infer CPI would create a risk of target
leakage.

The old `default` profile was retained for backward compatibility, while the
new experiments use `curvature`.

## 11. CPI prediction by the amortizer

During simulation, the true CPI is known and is used only as a training target.

CPI is normalized into `[-1, 1]`:

```text
z_CPI = 2 * (CPI - 400) / (1600 - 400) - 1
```

The predicted physical CPI is recovered with:

```text
predicted_CPI = 400 + (predicted_z_CPI + 1) / 2 * 1200
```

The network learns the statistical relationship between CPI and:

- cursor curvature;
- cursor speed;
- movement time;
- click accuracy;
- endpoint correction;
- cursor-to-target trajectories over multiple trials.

It does not directly calculate CPI from a supplied gain ratio.

## 12. Human-experiment integration

The same curvature extractor is used for simulated and human trajectories.

For each human trial:

1. raw cursor-event positions are collected;
2. the initial cursor position is prepended;
3. the cursor-only curvature extractor is applied;
4. all ten features are written to `trials.csv`.

This improves simulator-to-human consistency.

### Mousepad condition

A new `mousepad_condition` field was added with the values:

- `firm`;
- `soft`;
- `glass`;
- `unspecified`.

It is stored in:

- session metadata;
- raw mouse-event rows;
- frame rows;
- trial summaries;
- block-start and block-end events.

Surface friction may affect physical movement and correction behavior, so
recording it allows this factor to be controlled during analysis.

### Display consistency

The human experiment uses the same:

- 24-inch display profile;
- 1920 x 1080 resolution;
- physical display dimensions;
- 400-CPI reference;
- base gain of `4.357877685367455`.

## 13. CPI-curvature analysis tool

The new `src/scripts/analyze_cpi_curvature.py` script:

1. loads generated amortizer datasets;
2. extracts simulated CPI and the ten curvature features;
3. averages trials within each simulated parameter set;
4. calculates Pearson correlation;
5. calculates Spearman rank correlation;
6. reports p-values;
7. ranks features by absolute Spearman correlation;
8. writes `curvature_cpi_correlation.csv`.

Trials are averaged before correlation so repeated trials from one simulated
user are not treated as statistically independent observations.

The report is generated automatically before amortizer training when the
curvature pipeline is used.

## 14. Yonsei AI Data Center support

### RL Slurm launcher

A dedicated RL launcher was added with:

- RTX 3090 partition;
- one GPU;
- 64 CPU workers;
- 32 GB RAM;
- maximum three-day allocation;
- CUDA preflight validation;
- default 30-million-step training;
- checkpoint-resume support;
- configurable run names and orientation/task conditions.

### Dataset and amortizer launcher

A second Slurm launcher was added with:

- one RTX 3090;
- 64 CPUs;
- 128 GB RAM;
- curvature profile enabled by default;
- 20-Hz auxiliary trajectories;
- 131,072 training parameter sets;
- 64 trials per parameter set;
- 100 validation users;
- 256 validation trials per user;
- 200 amortizer iterations;
- 2,048 optimization steps per iteration;
- batch size 64;
- checkpoint and validation output every 10 iterations;
- automatic CPI-curvature report generation.

### Smoke-test mode

A lightweight smoke configuration was added:

- 32 parameter sets;
- 8 trials per parameter set;
- 8 validation users;
- 16 validation trials;
- 2 amortizer iterations;
- 32 optimization steps per iteration;
- checkpoint every iteration.

This validates dependencies, dataset generation, GPU access, feature
compatibility, training, and checkpoint creation before the expensive full
run.

### Pipeline submission

One submission script launches RL followed by dataset generation and amortizer
training with an `afterok` dependency. A second script supports `smoke` and
`full` modes for an already completed RL model.

## 15. Validation and tests

The expanded test suite verifies:

- exact 24-inch Full HD gain calibration;
- one-count-to-one-pixel mapping;
- linear CPI scaling;
- integer count quantization;
- sub-count residual accumulation;
- separation of virtual and executed residual state;
- scalar/vectorized mouse-mapping equivalence;
- 400-1600 CPI sampling in steps of 10;
- symmetric orientation sampling;
- zero-orientation and asymmetric control configurations;
- enlarged cursor-velocity ranges;
- agreement between human and RL count conversion;
- zero curvature for straight trajectories;
- expected total turn for circular arcs;
- stability across scale, translation, rotation, and sampling frequency;
- handling of duplicate and invalid points;
- absence of CPI and gain arguments in the curvature extractor;
- absence of hand, raw-distance, gain, and CPI fields in the curvature profile.

The complete local test suite passed:

```text
28 tests passed
```

A complete local mini-pipeline also passed:

- dataset generation;
- curvature report generation;
- amortizer training;
- parameter-recovery validation;
- checkpoint creation.

The Yonsei smoke pipeline completed successfully.

## 16. Training status

The completed Yonsei RL model is:

```text
pnc_rl_24fhd_cpi_26210DBCT
```

Validated artifacts include:

- `last/last_model.zip`;
- `best/best_model.zip`;
- `best/best_eval_summary.yaml`.

The RL directory is approximately 8.7 GB.

The curvature amortizer smoke test completed successfully, and the full
amortizer training was subsequently launched.

## 17. Important design decisions

### Why physical hand features were removed

Physical hand distances derived from raw counts require the true CPI:

```text
distance = counts * 0.0254 / CPI
```

Using this distance to predict CPI can leak the target value. The final
curvature profile therefore relies on observable cursor geometry.

### Why curvature is calculated before 20-Hz downsampling

A 20-Hz trajectory may contain relatively few points for a short movement.
Calculating curvature before temporal downsampling preserves the original path
shape.

### Why the path is resampled to 64 arc-length points

Direct curvature from a 1-kHz trajectory is sensitive to count-level jitter and
duplicate positions. Arc-length resampling creates a stable geometric
resolution and makes early, middle, and late movement comparisons consistent.

### Why the auxiliary trajectory remains at 20 Hz

Moving from 20 to 100 Hz would produce approximately five times more trajectory
points, storage, padding, and trajectory-encoder computation. Because the
engineered curvature already uses the full replay, 100 Hz was retained as a
possible future ablation rather than the main configuration.

## 18. Known current limitation

There is an iteration-label offset in validation output.

For example:

```text
checkpoint: pts/iter030.pt
validation PNG directory: results/iter029/
```

The checkpoint is correctly saved at iteration 30. The PNG directory uses the
previous value of the trainer's internal iteration counter because validation
is called before `self.iter` is updated.

This does not alter the trained model or metrics, but the validation-directory
label is shifted by one iteration.

## 19. Main modified or added files

### Mouse and CPI physics

- `src/agent/module_mouse.py`
- `src/agent/pnc_task.py`
- `src/utils/mouse_sensitivity.py`
- `src/config/pnc_env.yaml`
- `src/config/agent.yaml`

### Curvature and simulation summaries

- `src/utils/trajectory_features.py`
- `src/agent/pnc_simulator.py`
- `src/config/constant.py`
- `src/config/inference.yaml`

### Human experiment

- `src/experiment/async_logger.py`
- `src/experiment/config.yaml`
- `src/experiment/point_and_click_experiment.py`
- `src/experiment/web_experiment_server.py`

### Dataset and amortizer analysis

- `src/scripts/all_in_one_run.py`
- `src/scripts/analyze_cpi_curvature.py`

### Yonsei execution

- `run_bundle_yonsei.slurm`
- `run_bundle_amortizer_yonsei.slurm`
- `submit_bundle_pipeline_yonsei.sh`
- `submit_curvature_amortizer_yonsei.sh`

### Tests

- `tests/test_cpi_and_display.py`
- `tests/test_vectorized_math.py`
- `tests/test_trajectory_curvature.py`

### Documentation

- `RL_TRAINING_GUIDE.md`

## 20. Final summary

The original bundle modeled CPI primarily as a continuous gain effect. The
modified bundle now models a physical mouse sensor more faithfully through
integer counts, residual sub-count motion, physical display calibration,
CPI-dependent effective gain, and controlled orientation variability.

The inference system was redesigned around observable cursor-path geometry.
Ten robust curvature features are calculated before temporal downsampling and
are shared between simulated and human trajectories. Physical hand features
that could leak CPI were removed from the new inference profile.

The resulting pipeline can:

- train a CPI-aware RL pointing policy;
- generate large simulated datasets;
- quantify CPI-curvature relationships;
- train an amortizer to recover CPI and other user parameters;
- collect matching human experimental data;
- run reproducibly on the Yonsei AI Data Center.
