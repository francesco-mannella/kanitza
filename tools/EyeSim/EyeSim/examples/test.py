import EyeSim
import gymnasium as gym
import matplotlib.pyplot as plt


plt.ion()
_ = EyeSim

env = gym.make("EyeSim/EyeSim-v0")
env = env.unwrapped
env.init_world(world=0)
env.reset()
env.render(mode="human")
input()
