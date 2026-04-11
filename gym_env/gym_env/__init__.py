from importlib.metadata import entry_points
from gym.envs.registration import register

register(
    id = 'airsim-env-v0',
    # entry_point = 'gym_env.envs:AirsimGymEnv'
    entry_point='gym_env.envs.airsim_env:AirsimGymEnv'
)