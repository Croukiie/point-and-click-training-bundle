# v12 CPI-fix RL + summary-only shared-CPI amortizer

## What this run means

- RL checkpoint: `pnc_rl_v12_cpifix_26204ASOS`, checkpoint `best`.
- Amortizer input: summary-only; no trajectory sequence is passed to the network.
- Preprocessing frequency: 20 Hz.
- CPI inference: the three blocks are anonymous and their true CPI values are not model inputs. True CPI is used only after inference to evaluate recovery.
- Feature profile: `clean7` by default.
- Important: the archived RL was trained with `PnCEnvDefault`. Therefore CPI is hidden from the amortizer, but it was not hidden from the RL policy during RL training.

## 1. Upload from Windows PowerShell

Run from any PowerShell window:

```powershell
scp "$env:USERPROFILE\Downloads\pnc_rl_v12_cpifix_26204ASOS_best_bundle.tar.gz" `
  "C:\Users\elisa\Documents\Codex\2026-07-29\quelles-sont-les-diff-rences-entre\v12_summary_hidden_yonsei_update.tar.gz" `
  croukiie@165.132.142.207:~/point-and-click-training-bundle/
```

## 2. Install safely on Yonsei

Only the model directory is extracted from the RL archive, so its root-level requirements and old launch script cannot overwrite the current bundle.

```bash
cd ~/point-and-click-training-bundle

tar -xzf pnc_rl_v12_cpifix_26204ASOS_best_bundle.tar.gz \
  data/pnc_model/pnc_rl_v12_cpifix_26204ASOS

cp src/scripts/shared_cpi_2step_run.py \
  src/scripts/shared_cpi_2step_run.py.before_v12_summary
cp run_shared_cpi_2step_yonsei.slurm \
  run_shared_cpi_2step_yonsei.slurm.before_v12_summary

tar -xzf v12_summary_hidden_yonsei_update.tar.gz
chmod +x submit_v12_summary_hidden_yonsei.sh

.venv/bin/python -m compileall -q src/scripts/shared_cpi_2step_run.py
bash -n run_shared_cpi_2step_yonsei.slurm
bash -n submit_v12_summary_hidden_yonsei.sh

for file in \
  data/pnc_model/pnc_rl_v12_cpifix_26204ASOS/best/best_model.zip \
  data/pnc_model/pnc_rl_v12_cpifix_26204ASOS/env_info.yaml \
  data/pnc_model/pnc_rl_v12_cpifix_26204ASOS/train_config.yaml
do
  test -f "$file" && echo "OK: $file" || { echo "MISSING: $file"; exit 1; }
done
```

## 3. Smoke test

The new RL needs its own simulated dataset, so keep `REUSE_DATA=0`.

```bash
REUSE_DATA=0 RUN_TAG=v12_summary_hidden_smoke \
  bash submit_v12_summary_hidden_yonsei.sh smoke
```

Follow progress:

```bash
JOBID=$(cat .last_v12_summary_hidden_job)
squeue -j "$JOBID"
tail -F "slurm-pnc-shared-cpi-${JOBID}.out"
```

## 4. Full run after a successful smoke test

```bash
REUSE_DATA=0 RUN_TAG=v12_summary_hidden_full \
  bash submit_v12_summary_hidden_yonsei.sh full
```

For a retry using an already completed dataset with the same full settings, use `REUSE_DATA=1`.
