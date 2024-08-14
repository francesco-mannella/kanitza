import numpy as np
from scipy import interpolate
import gymnasium as gym
import EyeSim

env = gym.make('EyeSim-v0')


env.reset()
env.render()
input()

