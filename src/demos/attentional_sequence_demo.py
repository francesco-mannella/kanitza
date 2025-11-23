# %% IMPORTS

import EyeSim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from model.agent import Agent
from params import Parameters
from plotter import FoveaPlotter


_ = EyeSim

# This code is designed for simulating and visualizing an agent’s behavior in
# an environment, specifically focusing on its attention mechanisms

# %% MAIN LOOP AND VISUALIZATION
if __name__ == "__main__":

    # %%
    # Enable interactive mode and close any previously opened plots
    plt.ion()
    plt.close("all")

    # Set up the environment and agent
    seed = 15
    env = gym.make("EyeSim/EyeSim-v0")
    env = env.unwrapped
    env.set_seed(seed)

    params = Parameters()
    params.agent_sampling_precision = 1 - 1e-10
    params.attention_max_variance = 1
    params.attention_fixed_variance_prop = 0.3
    params.attention_center_distance_variance_prop = 0.7
    params.attention_center_distance_slope = 2
    agent = Agent(
        env,
        seed=seed,
        focus_params=params,
    )

    focuses = np.load("retina_poses.npy")
    mask = np.linalg.norm(focuses[:, 1:], axis=1).round(1) > 3
    focuses = focuses[mask]
    focuses[:, 1:] += env.retina_size // 2

    # %%
    focuses[:, 2] = 80 - focuses[:, 2]

    focuses[:, 1:] /= env.retina_size

    print(focuses)

    worlds = [
        "triangle",
        "square",
        "circle",
    ]

    # Run the simulation for a fixed number of episodes
    for episode in range(3):
        world_id = next(
            i for i, world in enumerate(env.world_labels) if world == worlds[episode]
        )

        object_params = {"pos": [40.0, 40.0], "rot": 0.5}

        env.init_world(world=world_id, object_params=object_params)
        _, env_info = env.reset()

        # Precompute some constants
        action = [0, 0]

        # Create a plotting object for the current episode
        plotter = FoveaPlotter(env, offline=False)

        for k in range(4):
            for c, center in enumerate(
                focuses[focuses[:, 0] == world_id, 1:][(0 if k == 0 else 1) :]
            ):
                # Set agent parameters based on the current attention center
                agent.set_parameters(center)

                # Simulate for a fixed number of time steps
                for time_step in range(2):

                    if time_step != 0:
                        agent.set_parameters([0.5, 0.5])

                    observation, *_ = env.step(action)

                    # the object is observable at the beginning
                    if not (k == 0 and c < 3):
                        observation["RETINA"] *= 0
                    action, saliency_map, salient_point, color_saliency = (
                        agent.get_action(observation)
                    )

                    # Update the plotter with the current saliency map and
                    # salient point
                    plotter.step(
                        color_saliency,
                        saliency_map,
                        salient_point,
                        agent.attentional_mask,
                    )
                    plt.pause(0.5)

        # Save the plot for the current episode as a gif
        gif_file = f"episode_{episode:04d}"
        plotter.close(gif_file)
