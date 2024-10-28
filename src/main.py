# %% IMPORTS
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import wandb
import signal
import sys, os, re
import torch

matplotlib.use('agg')

from model.agent import Agent, gaussian_mask
from plotter import FoveaPlotter, MapsPlotter
from model.offline_controller import OfflineController
from params import Parameters


def signal_handler(signum, frame):
    signal.signal(signum, signal.SIG_IGN)  # ignore additional signals
    wandb.finish()
    sys.exit(0)


class Logger:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, number):
        with open(self.filename, 'a') as file:
            file.write(str(number) + '\n')


# MAIN LOOP AND VISUALIZATION
if __name__ == '__main__':


    import argparse

    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the 'seed' argument
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Set the seed for random number generation.',
    )
    parser.add_argument('--decaying_speed', type=float, default=None,
                        help='Speed at which decay occurs')
    
    parser.add_argument('--local_decaying_speed', type=float, default=None,
                        help='Local speed at which decay occurs')

    # Parse the arguments
    args = parser.parse_args()
    
    # Create an instance of Parameters with default or custom values
    params = Parameters()

    # Access the seed value
    seed = args.seed

    params.decaying_speed = args.decaying_speed if args.decaying_speed is not None else params.decaying_speed
    params.local_decaying_speed = args.local_decaying_speed if args.local_decaying_speed is not None else params.local_decaying_speed
    params.init_name = "sim_s_{:}_ds_{:}_lds_{:}".format(
            re.sub("\.","",f"{seed:06d}"),
            re.sub("\.","",f"{params.decaying_speed:010.4f}"),
            re.sub("\.","",f"{params.local_decaying_speed:010.4f}"),
            )

    competence_log = Logger('comp')


    signal.signal(
        signal.SIGINT, signal_handler
    )  # register the signal with the signal handler first

    # Initialize Weights & Biases logging with project and entity names
    wandb.init(
        project=params.project_name,
        entity=params.entity_name,
        name=params.init_name,
    )

    # Enable matplotlib's interactive mode and close any existing plots
    plt.ion()
    plt.close('all')

    torch.manual_seed(seed)

    # Configure the environment and agent
    env = gym.make(params.env_name)
    env = env.unwrapped
    env.set_seed(seed)
    env.reset()

    agent = Agent(
        env, sampling_threshold=params.agent_sampling_threshold, seed=seed
    )

    file_path = 'off_control_store'

    # Check if the offline control data file exists
    if os.path.exists(file_path):
        # Load the offline controller from the file if it exists
        off_control = OfflineController.load(file_path, env, params, seed)
    else:
        # Initialize a new offline controller
        off_control = OfflineController(env, params, seed)
    
    # Initialize a plotting object 
    if params.plot_maps:
        maps_plotter = MapsPlotter(env, off_control, offline=True)

    for epoch in range(off_control.epoch, off_control.epoch + params.epochs):

        off_control.epoch = epoch
        off_control.reset_states()

        # Determine if the current epoch is a plotting epoch based on the interval
        is_plotting_epoch = epoch % params.plotting_epochs_interval == 0

        # Ensure the last epoch is always a plotting epoch
        is_plotting_epoch = is_plotting_epoch or epoch == params.epochs - 1


        # Execute the simulation for a specified number of episodes
        for episode in range(params.episodes):
            is_last_episode = episode == params.episodes - 1

            if params.plot_sim and is_last_episode and is_plotting_epoch:
                fovea_plotter = FoveaPlotter(env, offline=True)

            _, env_info = env.reset()

            # Precompute an action array initialized to zeros
            action = np.zeros(env.action_space.shape)

            # Create random mean values for the Gaussian masks (attention centers)
            attention_focuses = off_control.generate_attentional_input(
                params.focus_num
            )

            for focus_idx, focus in enumerate(attention_focuses):

                # Execute the steps within the focus time
                for time_step in range(params.focus_time):

                    # Configure agent parameters according to the current attention focus
                    if time_step == int(0.5 * params.focus_time):
                        agent.set_parameters(focus)

                    # Main cycle
                    observation, *_ = env.step(action)
                    action, saliency_map, salient_point = agent.get_action(
                        observation
                    )

                    # Update the fovea_plotter with the current saliency map and salient point
                    if (
                        params.plot_sim
                        and is_last_episode
                        and is_plotting_epoch
                    ):
                        fovea_plotter.step(
                            saliency_map, salient_point, agent.attentional_mask
                        )

                    state = dict(
                        vision=observation['FOVEA'],
                        action=action,
                        attention=focus,
                    )

                    off_control.record_states(
                        episode, focus_idx, time_step, state
                    )

            if params.plot_sim and is_last_episode and is_plotting_epoch:
                # Save the current episode's plot as a GIF file
                gif_file = f'sim_{epoch:04d}'
                fovea_plotter.close(gif_file)

                # Log the generated GIF file to Weights & Biases
                wandb.log(
                    {
                        'Simulations': wandb.Video(
                            f'{gif_file}.gif', format='gif'
                        ),
                    },
                    step=epoch,
                )

            print(f'Episode: {episode}, Epoch: {epoch}')

        competence_log(off_control.competence.detach().cpu().numpy())

        # Filter salient events
        off_control.filter_salient_states()

        # Update the offline controller's stored maps
        off_control.update_maps()

        if params.plot_maps:
            maps_plotter.step()
            if is_plotting_epoch and epoch > 0:
                # Save the current maps as a GIF file
                file = f'maps_{epoch:04d}'
                maps_plotter.close(file)
                # Log the generated GIF file to Weights & Biases if the file exists
                wandb.log(
                        {
                            'history': wandb.Image(f'{file}.gif'),
                            'last':wandb.Image(f'{file}.png')
                            }, 
                        step=epoch)
                maps_plotter = MapsPlotter(env, off_control, offline=True)

        # Save the OfflineController state to a file at the end of the loop
        off_control.save(file_path)

    
    # Conclude the Weights & Biases logging session
    wandb.finish()
