"""Registers the AirSim Gym environment with gym."""

from gym.envs.registration import register

register(
    id='airsim-env-v0',
    entry_point='env.airsim_env:AirsimGymEnv',
)
