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

| Path | Description |
| --- | --- |
| `gym_env/` | Gym environment: AirSim env, multirotor dynamics, curriculum tool |
| `scripts/` | Training / evaluation entry points, worker threads, GUI |
| `configs/` | Curriculum and test configs |
| `transformer-trajectory/` | Earlier experimental line (legacy, see its own README) |

## Usage

```bash
pip install -r requirements.txt
# start an AirSim simulator with a scene of your choice, then:
python scripts/train_scripts.py
python scripts/eval_scripts.py -model_path <model.zip> -eval_eps 20
```

`import gym_env` registers the environment as `airsim-env-v0`.
