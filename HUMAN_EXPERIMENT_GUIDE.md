# Point-and-click CPI experiment

The experiment mirrors the RL task geometry: a fixed circular target, a virtual
screen of 0.600 x 0.3375 m, Fitts IDs sampled uniformly from 2.5 to 7.2, a
3-second timeout, and the first click ending the trial. The same trial templates
are replayed at 400, 800, and 1600 CPI; their order is counterbalanced from the
participant ID.

## Important

You may develop and validate the program while RL is training. Do **not** collect
experimental participant data while RL training is running: CPU/GPU contention
could cause missed frames and make timing conditions incomparable.

The preferred collection backend uses Windows Raw Input and a custom cursor. Raw
mouse counts are captured at their native rate and converted with:

`virtual movement = raw counts * 0.0254 / CPI * 4.16`

Windows pointer acceleration therefore does not alter the experimental cursor.
The hardware CPI still has to be selected physically before each block.

For Mac/Linux pilots, the web version can fall back to browser Pointer Lock
movement. This is useful for testing and remote demonstrations, but it is not
equivalent to Windows Raw Input: the operating system or browser may already
have transformed the mouse deltas. These sessions are marked with
`input_backend = browser_pointer_lock` in the output files and should not be
mixed with final Windows Raw Input data.

## Commands

Configuration-only validation (safe during RL training):

```powershell
.\.venv\Scripts\python.exe -m src.experiment.point_and_click_experiment --validate-only --participant P001
```

Short windowed debugging is possible after the RL run is stopped:

```powershell
.\.venv\Scripts\python.exe -m src.experiment.point_and_click_experiment --windowed --participant TEST
```

Real collection:

```powershell
.\.venv\Scripts\python.exe -m src.experiment.point_and_click_experiment `
  --participant P001 --age 24 --dominant-hand right `
  --mouse-model "MODEL" --pointer-acceleration off
```

The recommended display profile is:

- 24 inch Full HD monitor;
- resolution: 1920 x 1080 px;
- refresh rate: 240 Hz;
- active display area used by the protocol: approximately 530 x 300 mm.

The web interface can still be launched on another display. In that case, it
shows and logs a warning because the physical display size may differ from the
recommended 24 inch profile. Do not change Windows sensitivity between blocks.

## Web version with fixed task window

For collection with the local HTML interface:

```powershell
.\.venv\Scripts\python.exe -m src.experiment.web_experiment_server
```

Then open the URL printed by the server, usually:

```text
http://127.0.0.1:8765/
```

The web setup form displays the recommended 24 inch Full HD profile values:

- active screen width: 53.0 cm;
- active screen height: 30.0 cm;
- resolution: 1920 x 1080 px;
- refresh rate: 240 Hz.

These fields define the fixed task window. They are saved in `session.json` and
repeated in `trials.csv` so that trajectories can be reconstructed in physical
screen units after collection. If the actual browser display is not Full HD, a
warning is also saved.

The task geometry remains the same as the RL task (`0.600 x 0.3375 m`). By
default, the web page renders a fixed 1920 x 1080 px task window. Fullscreen is
optional: when enabled, the point-and-click area fills the full browser screen
and the actual rendered size is logged for each trial. On non-Full-HD displays,
interpret the data with the logged warning in mind.

### Mac/Linux pilot mode

On Mac/Linux, run the web version with the browser Pointer Lock backend:

```bash
python -m src.experiment.web_experiment_server --input-backend browser_pointer_lock
```

On Windows, you can also force this pilot backend for comparison:

```powershell
.\.venv\Scripts\python.exe -m src.experiment.web_experiment_server --input-backend browser_pointer_lock
```

The native Python/Pygame version also has a cross-platform fallback for local UI
debugging:

```bash
python -m src.experiment.point_and_click_experiment --participant TEST --windowed --input-backend pygame_relative
```

This fallback samples Pygame relative movement once per frame. It is less precise
than both Windows Raw Input and the web Pointer Lock path, so treat it as
debug/pilot only.

## Output

Each session gets an independent folder under `data/human_experiment/`:

- `session.json`: participant, declared hardware, task settings and CPI order;
- `raw_mouse_events.csv`: every input event and reconstructed cursor state;
- `frames.csv`: every displayed frame and its timing;
- `trials.csv`: one result row per trial attempt;
- `experiment_events.jsonl`: target onsets, clicks, block boundaries and stops.

All per-trial, per-frame and per-input-event CSV files include `input_backend`
so Windows Raw Input, browser Pointer Lock and Pygame fallback sessions can be
filtered separately.

Velocity, acceleration, jerk, corrections and trajectory statistics should be
calculated after collection from the raw time series. They are intentionally not
calculated in the 240 Hz rendering loop.

## Pause and main menu controls

Both the native Python version and the local HTML version support pausing.

Native Python controls:

- `P` or `Escape`: open the pause menu;
- `Space`: resume from the pause menu;
- `M`: show the main instruction screen from the pause menu;
- `Q`: save the data collected so far and quit.

Web controls:

- click `Pause / menu`, or press `P` / `Escape`, to open the pause menu;
- `Resume experiment` continues from the same block and trial;
- `Show instructions` returns to the main instruction screen;
- `Save and stop` saves the partial session and then allows returning to the
  setup screen for a new participant without restarting the server.

If pause is requested during an active trial, that trial is saved as invalid with
the reason `participant_paused` and is repeated after resuming. Pausing between
trials does not create an invalid trial.
