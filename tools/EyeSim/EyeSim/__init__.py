"""EyeSim package: registers the "EyeSim/EyeSim-v0" gymnasium environment."""
from gymnasium.envs.registration import register


register(
    id="EyeSim/EyeSim-v0",
    entry_point="EyeSim.envs:EyeSimEnv",
)
