# Transformer Trajectory — legacy experiments

Earlier experimental line of this research: **SAC agents with different
observation encoders** (Transformer / CNN / LSTM / MLP) for mapless local
trajectory planning of a quadrotor in AirSim. Superseded by the root project
(see the root README); kept as an archive.

## Contents

| Path | Description |
| --- | --- |
| `gym_env/` | AirSim Gym environment (older version of the root one) |
| `configs/` | Ablation configs per policy variant (encoder x observation x reward); `need_edit_*` are unfinished drafts |
| `scripts/` | GUI train / eval entries, custom SB3 policies, trained checkpoint `model_sb3.zip` |
| `airsim_settings/` | AirSim settings for train / eval / init modes |
| `tools/` | Voxel-grid map generation, env / GPU / wandb smoke tests |

The ablations compare encoder architectures (Transformer token attention,
CNN, LSTM, plain MLP) and observation mixes (lidar vector, lidar image, goal,
ego state) under SAC.

## Dependencies

See `requirements.txt` (Python 3.8). The original development used a locally
vendored copy of stable-baselines3, which is not included here; the custom
policies live in `scripts/utils/custom_policy_sb3.py`.

## Migration notice

Since **2026-01-01** this line stopped using AirSim — hard to scale for
parallel execution — and the simulation stack migrates to **Isaac Sim 4.5 +
Isaac Lab 2.1.0** for better parallelization and long-term maintainability.
