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

        self.epoch = 0

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
        self.incompetence = 1 - torch.tanh(
            self.params.decaying_speed * self.competence
        )
        self.local_incompetence = 1 - torch.tanh(
            self.params.local_decaying_speed * self.competences
        )
        self.local_incompetence = self.local_incompetence.reshape(-1, 1)

        self.learnigrate_modulation = (
            self.params.learnigrate_modulation_baseline
            + (
                self.params.learnigrate_modulation
                * self.local_incompetence
                * self.incompetence
            )
        )
        self.neighborhood_modulation = (
            +self.params.neighborhood_modulation_baseline
            + (
                self.params.neighborhood_modulation
                * self.incompetence
                * self.local_incompetence
            )
        )
        self.neighborhood_modulation = self.neighborhood_modulation.reshape(
            -1, 1
        )

    def generate_attentional_input(self, focus_num):

        # radius = 0.4 + 0.1 * self.rng.rand(focus_num)
        # angle = 2 * np.pi * self.rng.rand(focus_num)
        # attentional_input = (
        #     np.stack([np.cos(angle), np.sin(angle)]) * radius.reshape(1, -1)
        # ).T + 0.5
        #
        attentional_input = self.rng.rand(focus_num, 2)

        return attentional_input

    def record_states(self, episode, focus, ts, state):

        self.visual_states[episode, focus, ts] = (
            state['vision'].ravel() / 255.0
        )
        self.action_states[episode, focus, ts] = state['action']
        self.attention_states[episode, focus, ts] = state['attention']

    def filter_salient_states(self):

        # Identify indices of states where the magnitude of saccades exceeds the threshold.
        filtered_states = np.linalg.norm(self.action_states, axis=-1)
        filtered_states = 1 * (filtered_states > self.params.saccade_threshold)
        self.filtered_idcs = np.stack(np.where(filtered_states))

    def update_maps(self):

        idcs = self.filtered_idcs
        idcs = idcs[
            :, (2 < idcs[2]) & (idcs[2] < (self.params.focus_time - 2))
        ]

        # Extract and reshape attention states to a tensor format suitable for neural operations
        preattention_states = self.attention_states[
            idcs[0], idcs[1], idcs[2] - 2
        ]
        preattention = torch.tensor(preattention_states).reshape(
            -1, self.params.attention_size
        )
        attention_states = self.attention_states[idcs[0], idcs[1], idcs[2] + 2]
        attention = torch.tensor(attention_states).reshape(
            -1, self.params.attention_size
        )

        # Extract and reshape visual condition states to a tensor (previous time step)
        visual_conditions = self.visual_states[idcs[0], idcs[1], idcs[2] - 2]
        visual_conditions = torch.tensor(visual_conditions).reshape(
            -1, self.params.visual_size
        )

        # Extract and reshape visual effect states to a tensor (current time step)
        visual_effects = self.visual_states[idcs[0], idcs[1], idcs[2] + 2]
        visual_effects = torch.tensor(visual_effects).reshape(
            -1, self.params.visual_size
        )

        # Run through attention mapping process with a modulation factor
        self.attention_map(
            attention, std=self.params.neighborhood_modulation_baseline
        )
        point_attention_representations = (
            self.attention_map.get_representation('point')
        )
        grid_attention_representations = self.attention_map.get_representation(
            'grid'
        )

        # Run visual conditions mapping with the same modulation factor
        self.visual_conditions_map(
            visual_conditions, std=self.params.neighborhood_modulation_baseline
        )
        point_visual_conditions_representations = (
            self.visual_conditions_map.get_representation('point')
        )
        grid_visual_conditions_representations = (
            self.visual_conditions_map.get_representation('grid')
        )

        # Run visual effects mapping similarly with modulation
        self.visual_effects_map(
            visual_effects, std=self.params.neighborhood_modulation_baseline
        )
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
            output=visual_conditions_output,
            std=self.visual_conditions_map.std,
            target=grid_attention_representations,
            learnigrate_modulation=self.learnigrate_modulation,
            target_std=1,
        )

        self.visual_effects_updater(
            output=visual_effects_output,
            std=self.visual_effects_map.std,
            target=grid_attention_representations,
            learnigrate_modulation=self.learnigrate_modulation,
            target_std=1,
        )
        self.attention_updater(
            output=attention_output,
            std=self.attention_map.std,
            target=grid_attention_representations,
            learnigrate_modulation=self.learnigrate_modulation,
            target_std=1,
        )

    def compute_matches(self):

        pve = self.representations['pve']
        pa = self.representations['pa']

        # Compute the difference between the tensors
        difference = pve - pa

        # Compute the norm over the last dimension
        matches = torch.norm(difference, dim=-1)

        return matches

    def get_action_from_condition(self, condition):

        condition = torch.tensor(condition.ravel().reshape(1, -1)) / 255.0

        self.visual_conditions_map(
            condition, self.params.neighborhood_modulation_baseline
        )
        condition_representation = (
            self.visual_conditions_map.get_representation()
        )
        attentional_focus = self.attention_map.backward(
            condition_representation,
            self.params.neighborhood_modulation_baseline,
        )
        return (
            attentional_focus.cpu().detach().numpy(),
            condition_representation.cpu().detach().numpy(),
        )

    def save(self, file_path):
        """
        Serialize and save the state of the OfflineController to a file.
        """
        # Collect the current states to save
        state = {
            'epoch': self.epoch,
            'competence': self.competence,
            'competences': self.competences,
            'local_incompetence': self.local_incompetence,
            'visual_conditions_map_state_dict': self.visual_conditions_map.state_dict(),
            'visual_conditions_updater_optimizer_state_dict': self.visual_conditions_updater.optimizer.state_dict(),
            'visual_effects_map_state_dict': self.visual_effects_map.state_dict(),
            'visual_effects_updater_optimizer_state_dict': self.visual_effects_updater.optimizer.state_dict(),
            'attention_map_state_dict': self.attention_map.state_dict(),
            'attention_updater_optimizer_state_dict': self.attention_updater.optimizer.state_dict(),
            'rng_state': self.rng.get_state(),
        }

        # Save the state to file
        torch.save(state, file_path)

    @staticmethod
    def load(file_path, env, params, seed=None):
        """
        Load and deserialize the state of the OfflineController from a file.
        """
        # Load the saved state
        state = torch.load(file_path, weights_only=False)

        # Initialize a new instance of OfflineController
        offline_controller = OfflineController(env, params, seed)

        # Restore the state
        offline_controller.epoch = state['epoch'] + 1
        offline_controller.competence = state['competence']
        offline_controller.competences = state['competences']
        offline_controller.local_incompetence = state['local_incompetence']
        offline_controller.visual_conditions_map.load_state_dict(
            state['visual_conditions_map_state_dict']
        )
        offline_controller.visual_conditions_updater.optimizer.load_state_dict(
            state['visual_conditions_updater_optimizer_state_dict']
        )
        offline_controller.visual_effects_map.load_state_dict(
            state['visual_effects_map_state_dict']
        )
        offline_controller.visual_effects_updater.optimizer.load_state_dict(
            state['visual_effects_updater_optimizer_state_dict']
        )
        offline_controller.attention_map.load_state_dict(
            state['attention_map_state_dict']
        )
        offline_controller.attention_updater.optimizer.load_state_dict(
            state['attention_updater_optimizer_state_dict']
        )
        offline_controller.rng.set_state(state['rng_state'])

        return offline_controller
