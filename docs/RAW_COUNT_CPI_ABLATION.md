# Raw-count CPI ablation

## Purpose

The current shared-CPI amortizers recover CPI in simulation but underestimate
and flatten CPI on human trials. The experiment already records lossless
Windows Raw Input values (`dx_counts`, `dy_counts`, and `dt_ns`) for every mouse
event. This ablation tests whether count-space timing and magnitude contain CPI
cues that are absent from the 20 Hz cursor summaries.

It does **not** treat the nominal block label (400, 800, or 1600 CPI) as an input.
Those values remain evaluation targets only.

## Matched models

Both amortizers use:

- the same corrected CPI-visible RL policy;
- the same simulated participant profiles and random seed;
- the same 20 Hz movement-only preprocessing;
- the same nine shared participant parameters;
- one block-specific CPI per anonymous block;
- no trajectory encoder and no curvature summaries.

The only difference is:

- `speedpath`: the existing 11 movement/task/speed/path summaries;
- `rawcount`: the same 11 summaries plus five raw-count summaries.

## Added features

Raw events are aggregated into exact 125 Hz bins from reaction end to the click,
with the click interval retained:

1. `raw_count_moving_bins`
2. `raw_count_mean_per_moving_bin`
3. `raw_count_p90_per_moving_bin`
4. `raw_count_max_per_bin`
5. `raw_count_path_per_second`

Simulation reconstructs the exact integer sensor counts from the replayed hand
trajectory using the sampled CPI, initial mouse orientation, rotation gains,
plan correction, quantization, and carried sub-count residual. Human data uses
the recorded Windows Raw Input counts. Both sources then use the same binning
and summary formulas.

## Leakage protection

The raw-count features stay in integer counts and seconds. The code never uses
`raw_dx_m`, `raw_dy_m`, `raw_path_length_m`, or any other physical-distance
quantity obtained by dividing counts by the declared CPI. Such values would
leak the label that the amortizer is intended to infer.

The experiment's `cpi` column is still used only to group and evaluate blocks.

## Interpretation limit

Raw counts alone do not directly measure physical CPI. CPI is counts per inch,
so a direct physical measurement also requires a known desk displacement. The
ablation can reveal CPI-correlated movement behavior, but it cannot turn an
unknown hand distance into a physical CPI measurement.

The existing physical calibration should therefore remain the ground-truth
check. The current measurements were approximately 399, 791, and 1541 CPI for
the nominal 400, 800, and 1600 settings.

## Yonsei launch

After installing the update in the bundle root:

```bash
RL_MODEL=pnc_rl_24fhd_cpi_visible_integrity_30m_integrity_visible_260819_151300 \
bash submit_rawcount_visible_cpi_ablation_yonsei.sh
```

This queues one new matched dataset, both smoke tests, then both 200-iteration
full amortizers. Follow the percentage progress with:

```bash
bash watch_rawcount_visible_cpi_ablation.sh --follow
```

The array mapping is `0=speedpath` and `1=rawcount`.

## Re-export existing human trials

No new experiment is required because raw counts are already present:

```powershell
python -m src.experiment.export_human_amortizer_data `
  --data-root data\human_experiment `
  --output-dir data\human_experiment\amortizer_ready\rawcount125 `
  --trajectory-hz 20 `
  --raw-count-hz 125 `
  --protocol-version target_radius_matched_v2
```
