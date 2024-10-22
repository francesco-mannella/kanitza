import numpy as np
import torch
from model.topological_maps import (
    TopologicalMap,
    STMUpdater,
)


class OfflineController:
    """
    OfflineController class manages the update of topological maps using
    sensory inputs from an environment.

    """

    def __init__(
        self,
        env,
        params,
        seed=None,
    ):
        """
        Initialize the OfflineController with the given environment.

        """

        # Save the environment reference
        self.env = env

        # save parameters reference
        self.params = params

        # Init random generator with seed
        self.seed = seed if seed is not None else 0
        self.rng = np.random.RandomState(self.seed)

        # Init visual map
        input_size = np.prod(env.observation_space['FOVEA'].sample().shape)
        output_size = params.maps_output_size
        self.visual_map = TopologicalMap(input_size, output_size)
        self.visual_updater = STMUpdater(
            self.visual_map, self.params.maps_learning_rate
        )

        # Init attention  map
        input_size = params.attention_size
        output_size = params.maps_output_size
        self.attention_map = TopologicalMap(input_size, output_size)
        self.attention_updater = STMUpdater(
            self.attention_map, self.params.maps_learning_rate
        )

        # Init Visual storage

        self.params.visual_size = np.prod(self.env.observation_space['FOVEA'].sample().shape) 
        self.visual_states = np.zeros(
            [
                self.params.episodes,
                self.params.focus_num,
                self.params.focus_time,
                self.params.visual_size,
            ]
        )

        # Init action storage
        self.action_states = np.zeros(
            [
                self.params.episodes,
                self.params.focus_num,
                self.params.focus_time,
                self.params.action_size,
            ]
        )

        # Init attention storage
        self.attention_states = np.zeros(
            [
                self.params.episodes,
                self.params.focus_num,
                self.params.focus_time,
                self.params.attention_size,
            ]
        )

    def set_hyperparams(self):
        self.visual_states *= 0
        self.action_states *= 0
        self.attention_states *= 0

    def generate_attentional_input(self, focus_num):

        return self.rng.rand(focus_num, 2)

    def record_states(self, episode, focus, ts, state):

        self.visual_states[episode, focus, ts] = state['vision'].ravel()
        self.action_states[episode, focus, ts] = state['action']
        self.attention_states[episode, focus, ts] = state['attention']

    def filter_salient_states(self):

        filtered_states = np.linalg.norm(self.action_states, axis=-1)
        filtered_states = 1 * (filtered_states > 10)
        self.filtered_idcs = np.stack(np.where(filtered_states))

    def update_maps(self):

        idcs = self.filtered_idcs

        # Reshape and convert attention states to tensor
        attention = torch.tensor(
            self.attention_states[idcs[0], idcs[1], idcs[2]]
        ).reshape(-1, self.params.attention_size)

        # Reshape and convert visual states to tensor
        visual = torch.tensor(
            self.visual_states[idcs[0], idcs[1], idcs[2]]
        ).reshape(-1, self.params.visual_size)


        vision_output = self.attention_map(attention, std=1)
        attention_output = self.attention_map(attention, std=1)
        goals = self.attention_map.get_representation('grid')

        self.visual_updater(vision_output, goals, 0.01)
        self.attention_updater(attention_output, goals, 0.01)


    def update_predicts(self):
        pass
