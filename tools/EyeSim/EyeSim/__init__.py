from gymnasium.envs.registration import register

register(
    id="EyeSim-v0",
    entry_point="EyeSim.envs:EyeSimEnv",
)

from EyeSim.envs import EyeSim_env
