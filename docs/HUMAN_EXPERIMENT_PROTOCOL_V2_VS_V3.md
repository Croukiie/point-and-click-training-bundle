# Human Experiment Protocols: `target_radius_matched_v2` versus `simulator_geometry_exact_v3`

## Purpose

This document explains the difference between the two human point-and-click experiment protocols used in the project. The change to `simulator_geometry_exact_v3` was introduced to remove unintended differences between the human experiment and the simulated RL tasks.

The main distinction is that `target_radius_matched_v2` matched the intended target-radius range, while `simulator_geometry_exact_v3` preserves the complete sampled task geometry: cursor start, target position, target radius, movement distance, and Fitts ID.

## Protocol comparison

| Property | `target_radius_matched_v2` | `simulator_geometry_exact_v3` |
|---|---|---|
| Start target | Disabled | Enabled |
| Initial cursor position | Position left by the previous trial | Independently sampled for every logical task |
| Dependency between consecutive trials | The endpoint of one trial affects the next trial's geometry | Each task has controlled geometry independent of the previous endpoint |
| Target position | Sampled, but used with the current cursor position | Exact sampled simulator-compatible target position |
| Target radius | Recomputed from the realized cursor-to-target distance and potentially clamped | Exact sampled radius is preserved without recomputation or clamping |
| Target-radius range | Intended to match the simulation range | Explicitly enforced between 1.25 and 7 mm |
| Fitts ID sampling | A rejected geometry could cause Fitts ID to be sampled again | Fitts ID is sampled once; only cursor and target positions are resampled after rejection |
| Realized Fitts ID | Could differ from the nominal value after radius recomputation or clamping | Preserved exactly for the task geometry |
| Simulation-to-human geometry | Approximate | Exact by construction |
| Target movement during aiming | Fixed target | Fixed target |

Both protocols present a stationary task target. The word "geometry" does not mean that the target moves during the pointing movement. Target locations vary between trials, but each target remains fixed while the participant moves and clicks.

## What happened in `target_radius_matched_v2`

The v2 protocol disabled the start target. A new trial therefore began at the virtual cursor position left by the previous trial.

The task generator had a nominal target position, target radius, and Fitts ID, but the realized cursor position was no longer necessarily the cursor position originally sampled with that target. The experiment recomputed the cursor-to-target distance and target radius at trial onset. It could also clamp the radius to keep the target inside the display.

Consequently, the nominal and realized tasks could differ. This matters because the participant's previous movement and CPI condition could indirectly affect the geometry of the next trial. CPI-related behavior could therefore become confounded with target distance, radius, or Fitts ID.

The existing Elisa and Hyeeun pilot sessions were collected with this protocol and should remain labelled `target_radius_matched_v2`.

## How `simulator_geometry_exact_v3` works

Each logical task is generated as a paired geometry containing:

- a sampled cursor starting position;
- a sampled task-target position;
- a target radius between 1.25 and 7 mm;
- the corresponding movement distance;
- a Fitts ID between 2.5 and 7.2.

The participant first sees a red start target at the sampled cursor starting position. After a successful start click, the program snaps the virtual cursor exactly to the center of that start target. The task target is then displayed with the exact sampled target position and radius.

The start-target click is a positioning stage. It is not exported as a measured point-and-click task. Only the following task movement and terminal click are used for the human-versus-model inference analysis.

This procedure ensures that the human task begins with the same geometry that was sampled for the simulation. The target radius, movement distance, and Fitts ID no longer depend on the previous trial's endpoint.

## Fitts ID preservation

The protocol uses

```text
ID = log2(D / W + 1)
```

where `D` is the cursor-to-target distance and `W` is the target diameter.

In v3, Fitts ID is sampled once for a task. If the sampled cursor and target positions do not produce a valid target within the screen and radius constraints, only the positions are sampled again. The Fitts ID remains fixed. This avoids biasing the realized ID distribution toward geometries that are easier to accept.

## Data compatibility

The two protocol versions must not be silently combined as if they represented the same experimental design.

- Existing v2 sessions remain useful as a pre-integrity baseline.
- For exact simulator-geometry studies, use `simulator_geometry_exact_v3`.
- The project deliberately switched the active local experiment back to
  `target_radius_matched_v2` on 24 August 2026; keep these new sessions labelled
  and analysed as v2 rather than mixing them with v3 sessions.
- Human export and inference should filter sessions by `protocol_version`.
- A clean evaluation of the corrected RL and amortizers requires newly collected v3 trials.

The protocol name describes human task geometry. The associated `movement_exact20_endpoint_clickonly_v3` dataset contract additionally specifies the common inference preprocessing: reaction removal before resampling, an exact 20 Hz grid, preservation of movement start, and inclusion of the exact terminal click sample.

## Current local readiness check

The local source was checked on 24 August 2026.

- Active protocol: `target_radius_matched_v2`.
- Start target: disabled; the cursor remains continuous between trials.
- Virtual display: 24-inch Full HD profile, 1920 x 1080.
- Configured target-radius range: 1.25 to 7 mm.
- Configured Fitts ID range: 2.5 to 7.2.
- Generated validation set: 143 task templates per CPI phase.
- Sampled target-radius range: 1.25 to 7 mm; v2 recomputes and constrains the
  realized radius from the cursor position at trial onset.
- Configuration validation: passed.
- Asynchronous data-logger check: passed.
- CPI setting method and per-block confirmation: required and persisted in every trial row.
- Complete unit-test suite: 58 tests passed.
- Local web-server smoke test with Windows Raw Input: HTTP 200, passed.

No blocking software error was found. The current local code is ready to start a new v2 pilot experiment.

## Participant-session checklist

Before collecting final data:

1. Use the same 24-inch 1920 x 1080 display described in the configuration, or record any deviation.
2. Use Chrome and keep the experiment PowerShell window open.
3. Use the independently measured effective Windows Raw Input rate. The current
   lab setup is standardized at 125 Hz even though the mouse driver reports
   1000 Hz; enter 125 Hz in the setup form and verify each physical mouse.
4. Disable Windows pointer acceleration and record the Windows pointer-speed setting for reproducibility.
5. Verify the physical CPI values before the session when possible.
6. Select how CPI will be changed in the setup form (physical mouse button, mouse software, or saved hardware profile).
7. Change the mouse CPI only when the experiment requests the next condition, then press `Confirm [CPI] CPI and start`. The method and confirmation are written into every trial row.
8. Point directly to the white target; there is no red positioning target in v2.
9. Do not mix v2 sessions with v3 sessions during export unless the comparison explicitly treats protocol as a separate condition.

## Launch command

From PowerShell in the bundle root:

```powershell
cd C:\Users\elisa\point-and-click-training-bundle
powershell -ExecutionPolicy Bypass -File .\run_human_experiment_windows.ps1
```

The experiment should then be available at:

```text
http://127.0.0.1:8765/
```

If port 8765 is already occupied, use another port:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_human_experiment_windows.ps1 -Port 8766
```

The resulting session metadata should report:

```text
protocol_version: target_radius_matched_v2
start_target_enabled: false
```
