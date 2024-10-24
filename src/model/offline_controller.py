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

        self.competence = torch.tensor(0.0)
        self.competences = torch.tensor(0.0)
        self.local_incompetence = torch.tensor(0.0)

        # Save the environment reference
        self.env = env

        # save parameters reference
        self.params = params

        # Init random generator with seed
        self.seed = seed if seed is not None else 0
        self.rng = np.random.RandomState(self.seed)

        # Init visual conditions  map
        input_size = np.prod(env.observation_space['FOVEA'].sample().shape)
        output_size = params.maps_output_size
        self.visual_conditions_map = TopologicalMap(input_size, output_size)
        self.visual_conditions_updater = STMUpdater(
            self.visual_conditions_map, self.params.maps_learning_rate
        )

        # Init visual effects  map
        input_size = np.prod(env.observation_space['FOVEA'].sample().shape)
        output_size = params.maps_output_size
        self.visual_effects_map = TopologicalMap(input_size, output_size)
        self.visual_effects_updater = STMUpdater(
            self.visual_effects_map, self.params.maps_learning_rate
        )

        # Init attention map
        input_size = params.attention_size
        output_size = params.maps_output_size
        self.attention_map = TopologicalMap(input_size, output_size)
        self.attention_updater = STMUpdater(
            self.attention_map, self.params.maps_learning_rate
        )

        # Init Visual storage

        self.params.visual_size = np.prod(
            self.env.observation_space['FOVEA'].sample().shape
        )
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

        self.set_hyperparams()

    def reset_states(self):
        self.visual_states *= 0
        self.action_states *= 0
        self.attention_states *= 0

    def set_hyperparams(self):
        self.incompetence = 1 - torch.tanh(self.params.decaying_speed * self.competence)
        self.local_incompetence = 1 - torch.tanh(self.params.local_decaying_speed * self.competences)
        self.local_incompetence = self.local_incompetence.reshape(-1, 1)
        
        self.learnigrate_modulation = (
            self.params.learnigrate_modulation 
            * self.local_incompetence
            * self.incompetence
        )
        self.neighborhood_modulation = (
            0.5 * np.sqrt(2.0)
            + self.params.neighborhood_modulation
            * self.incompetence
            * self.local_incompetence
        )
        self.neighborhood_modulation = self.neighborhood_modulation.reshape(
            -1, 1
        )

    def generate_attentional_input(self, focus_num):

        return self.rng.rand(focus_num, 2)

    def record_states(self, episode, focus, ts, state):

        self.visual_states[episode, focus, ts] = state['vision'].ravel()
        self.action_states[episode, focus, ts] = state['action']
        self.attention_states[episode, focus, ts] = state['attention']

    def filter_salient_states(self):

        # Identify indices of states where the magnitude of saccades exceeds the threshold.
        filtered_states = np.linalg.norm(self.action_states, axis=-1)
        filtered_states = 1 * (filtered_states > self.params.saccade_threshold)
        self.filtered_idcs = np.stack(np.where(filtered_states))

    def update_maps(self):

        idcs = self.filtered_idcs

        # Extract and reshape attention states to a tensor format suitable for neural operations
        attention_states = self.attention_states[idcs[0], idcs[1], idcs[2]]
        attention = torch.tensor(attention_states).reshape(
            -1, self.params.attention_size
        )

        # Extract and reshape visual condition states to a tensor (previous time step)
        visual_conditions = self.visual_states[idcs[0], idcs[1], idcs[2] - 1]
        visual_conditions = torch.tensor(visual_conditions).reshape(
            -1, self.params.visual_size
        )

        # Extract and reshape visual effect states to a tensor (current time step)
        visual_effects = self.visual_states[idcs[0], idcs[1], idcs[2]]
        visual_effects = torch.tensor(visual_effects).reshape(
            -1, self.params.visual_size
        )

        # Run through attention mapping process with a modulation factor
        self.attention_map(attention, std=1)
        point_attention_representations = (
            self.attention_map.get_representation('point')
        )
        grid_attention_representations = self.attention_map.get_representation(
            'grid'
        )

        # Run visual conditions mapping with the same modulation factor
        self.visual_conditions_map(visual_conditions, std=1)
        point_visual_conditions_representations = (
            self.visual_conditions_map.get_representation('point')
        )
        grid_visual_conditions_representations = (
            self.visual_conditions_map.get_representation('grid')
        )

        # Run visual effects mapping similarly with modulation
        self.visual_effects_map(visual_effects, std=1)
        point_visual_effects_representations = (
            self.visual_effects_map.get_representation('point')
        )
        grid_visual_effects_representations = (
            self.visual_effects_map.get_representation('grid')
        )

        # Store various representation types for further processing
        self.representations = {
            'pvc': point_visual_conditions_representations,  # Point level: Visual Conditions
            'pve': point_visual_effects_representations,  # Point level: Visual Effects
            'pa': point_attention_representations,  # Point level: Attention
            'gvc': grid_visual_conditions_representations,  # Grid level: Visual Conditions
            'gve': grid_visual_effects_representations,  # Grid level: Visual Effects
            'ga': grid_attention_representations,  # Grid level: Attention
        }

        # Compute matching scores for current state projections
        self.matches = self.compute_matches()

        # Compute competences based on matches using a Gaussian-like decay function
        s = 3.0
        self.competences = np.exp(-(s**-2) * self.matches**2)
        self.competence = self.competences.mean()

        # Update offline controller hyperparameters based on the episodes
        self.set_hyperparams()

        attention_output = self.attention_map(
            attention, std=self.neighborhood_modulation
        )
        visual_conditions_output = self.visual_conditions_map(
            visual_conditions, std=self.neighborhood_modulation
        )
        visual_effects_output = self.visual_effects_map(
            visual_effects, std=self.neighborhood_modulation
        )

        # Reshape competences and use them to update conditions, effects, and attention
        self.visual_conditions_updater(
            visual_conditions_output,
            grid_attention_representations * self.learnigrate_modulation,
            1,
        )
        self.visual_effects_updater(
            visual_effects_output,
            grid_attention_representations * self.learnigrate_modulation,
            1,
        )
        self.attention_updater(
            attention_output,
            grid_attention_representations * self.learnigrate_modulation,
            1,
        )

    def compute_matches(self):

        pve = self.representations['pve']
        pa = self.representations['pa']

        # Compute the difference between the tensors
        difference = pve - pa

        # Compute the norm over the last dimension
        matches = torch.norm(difference, dim=-1)

        return matches
