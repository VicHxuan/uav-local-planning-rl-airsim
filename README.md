# uav-local-planning-rl-airsim

Mapless local planning for quadrotor UAVs with reinforcement learning on **AirSim**.

RL agents (SAC / TD3 / PPO via Stable-Baselines3) learn local motion commands
directly from onboard observations (depth / lidar, goal and ego state) without
any global map. Training and evaluation run with a live PyQt5 GUI, and a
curriculum manager schedules task difficulty during training.

> **Status (2026-01):** AirSim is deprecated for this project (poor parallel
> scalability). New work targets Isaac Sim 4.5 + Isaac Lab 2.1.0 in a separate
> repository. This repository is the AirSim-era archive.

## Layout

```
src/               importable package code (entry scripts add it to sys.path)
  env/             AirSim gym environment, multirotor dynamics, curriculum manager
  policies/        SB3 feature extractors and observation encoders (MLP / LSTM / CNN / Transformer)
  app/             GUI layer: training / evaluation worker threads, monitor UI
scripts/           thin entry points: train.py, evaluate.py, infer_min.py
configs/           test.ini, curriculum.yaml, and experiments/ ablation configs
models/           trained checkpoint (model_sb3.zip)
tools/             AirSim connection check, map generation, env smoke tests
airsim_settings/   AirSim client settings (train / eval variants)
```

## Usage

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
# start an AirSim simulator with a scene of your choice, then:
python scripts/train.py                                   # GUI training run (configs/test.ini)
python scripts/evaluate.py -model_path <model.zip> -eval_eps 20
python scripts/infer_min.py --model models/model_sb3.zip  # REPL inference without a simulator
```

The entry scripts put `src/` on `sys.path`; `import env` registers the
environment with gym as `airsim-env-v0`.
