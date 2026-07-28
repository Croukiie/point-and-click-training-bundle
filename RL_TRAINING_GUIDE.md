# RL Training Guide

## Setup

Python 3.9 또는 3.10 환경을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-rl.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Full Training

The task uses the physical dimensions of a 24-inch 16:9 display
(`0.5313 x 0.2989 m`). CPI is a policy parameter sampled from 400 to 1600 in
steps of 10. The historical mouse gain of 4.16 is calibrated at 400 CPI, so
the effective gain is `4.16 * CPI / 400`. Every trial independently samples
its initial physical-mouse orientation from a continuous uniform distribution
between `-30` and `+20` degrees.

기본 full training은 `src/config/rl_config.yaml`의 `simple` 설정을 사용합니다.
현재 기본 step 수는 `10,000,000`입니다.

```bash
bash scripts/run_full_rl.sh
```

동일한 명령을 직접 쓰면 다음과 같습니다.

```bash
python -m src.rl.sac_trainer \
  --num_cpu 8 \
  --env_class_type PnCEnvDefault \
  --agent_type classic \
  --pnc_type default \
  --rl_setting simple \
  --model_name_prefix pnc_mouse_full
```

`--num_cpu`는 rollout environment 병렬 개수입니다. 사용하는 머신의 물리 코어 수를 넘기지 않는 것을 권장합니다.

## Device

기본값은 `device: auto`입니다.

- CUDA가 있으면 CUDA 사용
- Mac MPS가 있으면 MPS 사용
- 둘 다 없으면 CPU 사용

강제로 지정하려면:

```bash
python -m src.rl.sac_trainer ... --device cuda
python -m src.rl.sac_trainer ... --device mps
python -m src.rl.sac_trainer ... --device cpu
```

## Output

결과는 `data/pnc_model/<run_name>/` 아래에 저장됩니다.

```text
data/pnc_model/<run_name>/
  env_info.yaml
  train_config.yaml
  tb_log/
  checkpoint/
  best/
  evaluation/
```

Each evaluation now records the normalized/actual user parameters, task
conditions, reward, success, click distance, episode length, and motor effort.

```text
evaluation/evaluations.npz
evaluation/parameters/episodes.jsonl
evaluation/parameters/summaries.jsonl
best/best_model.zip
best/visualizer_best_params.yaml
best/best_eval_summary.yaml
best/best_eval_episodes.json
```

`best/visualizer_best_params.yaml` belongs to the same evaluation that produced
`best_model.zip`. The visualizer loads this parameter profile automatically.

## TensorBoard

```bash
tensorboard --logdir data/pnc_model --host 127.0.0.1 --port 6006
```

브라우저에서 엽니다.

```text
http://127.0.0.1:6006/
```

## Resume

checkpoint 예시:

```text
data/pnc_model/<run_name>/checkpoint/rl_model_5000000_steps.zip
data/pnc_model/<run_name>/checkpoint/rl_model_replay_buffer_5000000_steps.pkl
```

재개:

```bash
python -m src.rl.sac_trainer \
  --num_cpu 8 \
  --env_class_type PnCEnvDefault \
  --agent_type classic \
  --pnc_type default \
  --rl_setting simple \
  --model_name_prefix <new_or_same_run_name> \
  --load_model data/pnc_model/<run_name>/checkpoint/rl_model_5000000_steps.zip \
  --load_replay_buffer data/pnc_model/<run_name>/checkpoint/rl_model_replay_buffer_5000000_steps.pkl
```

## Main Files

```text
src/agent/pnc_agent.py       Gymnasium environment and observation construction
src/agent/pnc_task.py        point-and-click task and mouse-hand dynamics
src/agent/module_mouse.py    mouse rotation/sensor mapping
src/rl/sac_trainer.py        SAC training entrypoint
src/rl/sac_policy.py         modulated SAC actor/critic
src/config/agent.yaml        observation list and agent trait ranges
src/config/rl_config.yaml    SAC hyperparameters
```

## Policy Visualization

The interactive visualizer provides parameter sliders, Run/Pause/Reset
controls, a monitor view (target and cursor), a desk view (hand and physical
mouse), distance/rotation plots, and episode metrics. When `--model` is
omitted, it uses the newest `best_model.zip` found below `data/pnc_model`.

```powershell
python -m src.rl.visualize_policy
```

Save the replay as a GIF without opening a window:

```powershell
python -m src.rl.visualize_policy --save data/pnc_model/policy_visualization.gif --no-show
```

Use a specific model:

```powershell
python -m src.rl.visualize_policy --model <run_name>
```

`--model <run_name>` automatically opens that run's latest saved model
(`last/last_model.zip`, or the newest numbered checkpoint if the final model has
not been exported yet). It also loads `visualizer_last_params.yaml` from that run
as the initial slider profile. Use `--ckpt best --slider-profile best` to inspect
the best checkpoint/profile instead, or `--slider-profile midpoint` to start from
range midpoints.
To visualize a numbered checkpoint instead:

```powershell
python -m src.rl.visualize_policy --model <run_name> --ckpt 10000000
```
## Amortized inference from the converged mouse policy

The legacy inference simulator is configured as `mouse_converge_best` in
`src/config/inference.yaml`. Its saved checkpoint predates CPI and still
contains nine policy parameters. Train a new CPI-aware run with the complete
pipeline below before generating a CPI dataset.

Install the inference-only dependencies once:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-inference.txt
```

Run a small smoke test before generating the full dataset:

```powershell
.\.venv\Scripts\python.exe -m src.inference.pnc_dataset `
  --simul_config mouse_converge_best `
  --n_param 2 --n_trial 2 `
  --valid_users 2 --valid_trials 2 `
  --num_cpu 1
```

Generate the training and validation simulations:

```powershell
.\.venv\Scripts\python.exe -m src.inference.pnc_dataset `
  --simul_config mouse_converge_best `
  --n_param 131072 --n_trial 64 `
  --valid_users 100 --valid_trials 256 `
  --num_cpu 8
```

Then train the point-estimation amortizer:

```powershell
.\.venv\Scripts\python.exe -m src.inference.pnc_infer_trainer `
  --name PTE_mouse_converge_best `
  --simul_config mouse_converge_best `
  --train_config default --data_config default `
  --n_iter 100 --step_per_iter 2048 `
  --batch_sz 64 --n_trial 64 --save_freq 10
```

## Complete CPI pipeline

The all-in-one runner trains the RL policy, generates both datasets, and then
trains the amortizer. By default it generates `2^17 = 131072` training
parameter sets, with 64 trials per set:

```powershell
.\.venv\Scripts\python.exe -m src.scripts.all_in_one_run
```

The CPI amortizer uses both per-trial summaries and 20 Hz trajectories. Its
trajectory encoder receives time deltas, cursor position, physical hand
position, and cursor-to-target relative position. Keeping cursor and hand
motion together makes their displacement ratio, and therefore CPI, observable.

To reuse an existing RL run:

```powershell
.\.venv\Scripts\python.exe -m src.scripts.all_in_one_run `
  --skip-rl --rl-model <run_name> --rl-checkpoint best
```

## Human CPI experiment

Install the dedicated UI dependencies:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-experiment.txt
```

The protocol uses the same 400-CPI reference gain and exact 24-inch 16:9
surface as the RL environment. Start the local browser experiment with:

```powershell
.\.venv\Scripts\python.exe -m src.experiment.web_experiment_server
```
