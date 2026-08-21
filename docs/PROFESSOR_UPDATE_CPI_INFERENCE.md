# Progress report: CPI inference from point-and-click behavior

## Objective

Our goal is to infer physical mouse CPI and participant-specific mouse-control parameters from point-and-click behavior using a reinforcement-learning simulator and amortized inference. The main scientific question is whether CPI leaves identifiable trajectory, speed, and curvature signatures that transfer from simulation to human data.

## Work completed

We first standardized the experimental setup for a 24-inch Full-HD display and recorded Windows Raw Input with pointer acceleration disabled and Windows pointer speed set to 6/11. Human data were collected at 400, 800, and 1600 CPI with a fixed software cursor mapping, so changing the physical mouse CPI changed the effective cursor speed. We currently have 767 valid measured trials from two participants.

We then implemented and tested the following changes:

1. Matched the human and simulation target-radius range and recalibrated the simulator against the human task geometry.
2. Corrected the task sampler so that Fitts ID is sampled once and only positions are resampled when a task is invalid. The realized simulated ID distribution passed the uniformity audit.
3. Corrected trajectory replay so that CPI, mouse rotation gains X/Y, and mouse plan correction are copied exactly from the original RL rollout.
4. Standardized trajectory processing at 20 Hz: reaction removal before resampling, preservation of the movement start, and inclusion of the click endpoint.
5. Added ten path/curvature summaries and progressively tested reduced feature sets: full curvature, CLEAN7, CORE3, and speed/path only.
6. Tested trajectory and summary-only amortizers, as well as RL policies with CPI visible and CPI hidden.
7. Implemented shared two-step inference. The nine non-CPI participant parameters are shared across the three anonymous blocks, while a separate CPI is inferred for each block. True CPI and the block labels are not supplied as amortizer input features.
8. Trained the final corrected visible-CPI RL policy for 30 million steps and generated 8,388,480 training trials. Two matched summary-only amortizers were then trained for 200 iterations on the same simulated dataset and seed: speed/path only and speed/path plus three CORE3 curvature features.

## CPI bias audit

We explicitly checked whether the ground-truth CPI distribution or its use in the pipeline was biased.

- The simulated training set contains 131,070 independently sampled CPI parameter sets on the discrete range 400--1600 CPI in steps of 10. Its mean is 999.76 CPI, and its distribution is statistically consistent with uniform sampling (chi-square p = 0.485). There is no block-position effect (p = 0.832).
- The simulated validation set has a mean of 975 CPI across 300 conditions. Monte Carlo testing shows that this deviation is compatible with ordinary sampling variation (p = 0.218), and its distribution is also compatible with uniform sampling (p = 0.426).
- The human dataset is balanced: 256 trials at 400 CPI, 256 at 800 CPI, and 255 at 1600 CPI. Every retained trial was recorded with CPI confirmation.
- Target radius, target distance, and realized Fitts ID have virtually identical distributions across the three human CPI conditions (all ANOVA p-values above 0.998). Task difficulty therefore does not explain the CPI result.
- Code and saved model configurations confirm that true CPI is not included in the amortizer summary features and that anonymous block labels do not encode CPI.

We therefore found no evidence that the simulated CPI targets, human task allocation, or amortizer inputs are biased toward a particular CPI.

The hardware settings were subsequently checked using five 20 cm Raw Input measurements per CPI. The measured means were 399.2 CPI for the nominal 400 setting (-0.2%), 791.0 CPI for 800 (-1.1%), and 1541.5 CPI for 1600 (-3.7%). The first two settings are extremely close to nominal, and the 1600 setting is slightly low but still within 5%. This small physical deviation cannot explain why the amortizer predicts approximately 489--556 CPI for the 1600 block. The remaining experimental-design limitation is that, with only two participants, the two block orders do not fully counterbalance learning, fatigue, or adaptation effects. A complete Latin-square block design is recommended for the next data collection.

## Current results

Both amortizers recover CPI accurately in simulated validation data. At 64 trials per block, simulated predictive R2 is 0.973 for speed/path and 0.979 for CORE3, with CPI slopes of 0.949 and 0.964 respectively.

However, human CPI recovery remains strongly compressed:

| Model | Mean inferred CPI at true 400 | At true 800 | At true 1600 |
|---|---:|---:|---:|
| Speed/path | 447 | 462 | 489 |
| CORE3 | 525 | 512 | 556 |

The human CPI slopes are only about 0.03--0.04 instead of 1.0. At true 1600 CPI, the average bias is approximately -1111 CPI for speed/path and -1044 CPI for CORE3. By contrast, the corresponding high-CPI bias in simulated validation is only about -55 and -44 CPI.

Re-exporting the human data with the fully aligned 20 Hz preprocessing changed the estimates only marginally. This rules out the previous reaction/resampling/endpoint mismatch as the main cause.

The strongest remaining problem is the simulation-to-human behavioral gap. The simulated validation hit rate is approximately 41%, whereas the human hit rate is 99.5%. Human movements are also longer and slower, with different speed, path-length, and curvature distributions. The amortizer therefore learns CPI cues that are highly predictive within the simulator but weak, shifted, or absent in human behavior. Adding CORE3 improves simulated recovery slightly but does not restore human CPI ordering.

## Interpretation and proposed next steps

The evidence currently supports estimator domain bias rather than biased CPI labels: the true CPI sampling is sound, but the human observations lie in a behavioral domain that the amortizer did not learn to interpret correctly.

Our proposed next steps are:

1. Collect additional participants with a fully counterbalanced CPI block order and the corrected v3 task protocol.
2. Recalibrate or retrain the simulator/RL policy to match human hit rate, movement time, click error, maximum speed, path length, and curvature distributions before training another amortizer.
3. Repeat a strictly matched ablation using speed/path only versus speed/path plus turn energy and late-turn curvature. Early-turn curvature should be removed initially because its human mean is approximately twelve times the simulated mean.
4. Re-evaluate simulated recovery and human CPI ordering only after the simulation-to-human feature gap has been reduced.

At this stage, the inferred human CPI values should be treated as diagnostic model outputs, not calibrated estimates of physical CPI.
