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

    Attributes:
        base_learning_rate (float): Base learning rate for updates.
        learning_rate_range (float): Range variation for the adaptive learning rate.
        learning_rate (float): Current adaptive learning rate.
        base_std (float): Base standard deviation for sensory input noise.
        std_range (float): Range variation for the adaptive standard deviation.
        std (float): Current adaptive standard deviation.
        env: Environment providing sensory inputs.
        fovea_map: Topological map for fovea sensory input.
        fovea_updater: STMUpdater instance for updating the fovea map.
        fovea_inputs (torch.Tensor): Preallocated tensor for batch of fovea inputs.
        fovea_grid_representations (torch.Tensor): Preallocated tensor for fovea grid representations.
        fovea_point_representations (torch.Tensor): Preallocated tensor for fovea point representations.
        fovea_input_counter (int): Counter for tracking the number of stored fovea inputs.
    """

    def __init__(
        self, env, input_batch_size=20000, maps_output_size=(10 * 10),
        base_learning_rate=0.001, learning_rate_range=0.05, base_std_factor=0.5,
        std_range=2.0, attentional_input_size=2, attentional_update_lr=0.1,
        attentional_weights_path='attentional_weights'
    ):
        """
        Initialize the OfflineController with the given environment.

        Args:
            env: An environment object which has an observation_space
                 dictionary with a 'FOVEA' key.
            input_batch_size (int): Batch size for input processing.
            maps_output_size (int): Size of the output map representation.
            base_learning_rate (float): The base learning rate.
            learning_rate_range (float): The range for the learning rate.
            base_std_factor (float): Base factor for standard deviation calculation.
            std_range (float): The range for the standard deviation.
            attentional_input_size (int): Size of the attentional input.
            attentional_update_lr (float): Learning rate for attentional updating.
            attentional_weights_path (str): Path to the attentional weights.
        """

        # Initialize input batch size and map output size
        self.input_batch_size = input_batch_size  # Batch size for input processing
        self.maps_output_size = maps_output_size  # Size of the output map representation
        self.batch_shape = (self.input_batch_size, self.maps_output_size)

        # Initialize learning rate and standard deviation parameters
        self.base_learning_rate = base_learning_rate
        self.learning_rate_range = learning_rate_range
        self.base_std = base_std_factor * np.sqrt(2)
        self.std_range = std_range

        self.set_hyperparams(0)

        # Save the environment reference
        self.env = env

        # Get observation space sample shape and calculate the input size
        fovea_sample = self.env.observation_space['FOVEA'].sample()
        self.fovea_input_size = fovea_sample.size

        # Initialize TopologicalMap and STMUpdater
        self.fovea_map = TopologicalMap(
            self.fovea_input_size, self.maps_output_size
        )
        self.fovea_updater = STMUpdater(self.fovea_map, self.learning_rate)

        # Preallocate tensors for batch processing
        self.fovea_inputs = torch.zeros(
            (self.input_batch_size, self.fovea_input_size)
        )
        self.fovea_positions = torch.zeros((self.input_batch_size, 2))
        self.fovea_episodes = torch.zeros((self.input_batch_size, 1))
        self.fovea_grid_representations = torch.zeros(self.batch_shape)
        self.fovea_point_representations = torch.zeros(
            (self.input_batch_size, 2)
        )

        # Initialize counter for fovea inputs
        self.fovea_input_counter = 0

        # Attentional input
        self.attentional_input_size = attentional_input_size

        # Initialize TopologicalMap and STMUpdater
        self.attentional_map = TopologicalMap(
            self.attentional_input_size,
            self.maps_output_size,
            parameters=torch.tensor(torch.load(attentional_weights_path).clone()),
        )
        self.attentional_updater = STMUpdater(self.attentional_map, attentional_update_lr)

        # Preallocate tensors for batch processing
        self.attentional_inputs = torch.zeros(
            (self.input_batch_size, self.attentional_input_size)
        )
        self.attentional_grid_representations = torch.zeros(self.batch_shape)
        self.attentional_point_representations = torch.zeros(
            (self.input_batch_size, 2)
        )

        # Initialize counter for attentional inputs
        self.attentional_input_counter = 0

    def set_hyperparams(self, competence):
        """
        Set the adaptive hyperparameters (learning rate and standard deviation)
        based on competence.

        Args:
            competence (float): A value indicating the current competence level.
        """
        exp_factor = 1 - competence

        self.learning_rate = (
            self.base_learning_rate + self.learning_rate_range * exp_factor
        )
        self.std = self.base_std + self.std_range * exp_factor

    def generate_attentional_input(self, num_focuses):

        points = 10*torch.rand(num_focuses, 2)

        attentional_inputs = self.attentional_map.backward(points, 0.5*np.sqrt(2))
        return attentional_inputs.cpu().detach().numpy()

    def store_attentional_input(self, attentional_input):
        """
        Stores and processes attentional input.

        Args:
            attentional_input (np.ndarray): The input coordinates in the retina.

        TODO:
            The current encoding of attentional representations is FAKE
            and needs proper implementation.
        """

        # Store the attentional input at the current counter index
        index = self.attentional_input_counter

        self.attentional_map(torch.tensor(attentional_input), self.std)

        self.attentional_inputs[index] += attentional_input

        # Create attentional grid representation using radial function
        self.attentional_grid_representations[
            index
        ] += self.attentional_map.get_representation('grid').reshape(-1)

        # Store the  point representation
        self.attentional_point_representations[
            index
        ] += self.attentional_map.get_representation('point').reshape(-1)


        # Increment the input counter
        self.attentional_input_counter += 1

    def store_episode_and_fovea_input(self, episode):
        """
        Store the foveal input data and extract corresponding representations.

        Args:
            episode (int): The current episode number.
        """
        # Flatten and normalize the fovea input from the environment observation
        fovea_input = self.env.observation['FOVEA'].ravel() / 255.0
        fovea_tensor = torch.tensor(fovea_input).unsqueeze(0)

        # Obtain the fovea output using the fovea map with standard deviation
        fovea_output = self.fovea_map(fovea_tensor, self.std)

        # Update the fovea datasets with the current episode data
        index = self.fovea_input_counter
        self.fovea_episodes[index] += episode
        self.fovea_positions[index] += self.env.retina_sim_pos
        self.fovea_inputs[index] += fovea_input
        self.fovea_grid_representations[
            index
        ] += self.fovea_map.get_representation('grid').ravel()
        self.fovea_point_representations[
            index
        ] += self.fovea_map.get_representation('point').ravel()

        # Increment the input counter
        self.fovea_input_counter += 1

    def update_maps(self):
        """
        Update the topological maps using stored fovea inputs and reset the
        preallocated tensors.
        """

        positions = self.fovea_positions[: self.fovea_input_counter]
        episodes = self.fovea_episodes[: self.fovea_input_counter]
        

        # Compute mask for distances within the same episode and distances vector
        same_episode_mask = episodes[:-1] == episodes[1:]

        # Calculate distances for consecutive positions and handle cross-episode distances
        consecutive_diffs = positions[1:] - positions[:-1]
        all_distances = torch.norm(consecutive_diffs, dim=1)
        all_distances[~same_episode_mask.flatten()] = -1

        # Concatenate an initial zero distance
        distances = torch.cat([torch.tensor([0]), all_distances])

        # Add 0 distances for indices greater than self.fovea_input_counter
        if self.fovea_input_counter < len(self.fovea_positions):
            zero_pad_length = (
                len(self.fovea_positions) - self.fovea_input_counter
            )
            distances = torch.cat([distances, torch.zeros(zero_pad_length)])

        idcs = distances > 4.0

        # Slicing the tensor to only include relevant fovea outputs
        inputs = self.fovea_inputs[idcs]
        targets = self.attentional_grid_representations[idcs]
        fovea_outputs = self.fovea_map(inputs, self.std)

        np.save('inputs', inputs.cpu().detach().numpy())
        # Updating the fovea map using the extracted outputs
        self.fovea_updater(fovea_outputs, targets, self.learning_rate)

        self.update_counter = getattr(self, 'update_counter', 0)
        np.save(f'inputs_{self.update_counter:03d}', inputs)
        self.update_counter += 1

        # Reset preallocated tensors to zero in-place for memory efficiency
        for tensor in [
            self.fovea_inputs,
            self.fovea_positions,
            self.fovea_episodes,
            self.fovea_grid_representations,
            self.fovea_point_representations,
            self.attentional_inputs,
            self.attentional_grid_representations,
            self.attentional_point_representations,
        ]:
            tensor.zero_()
        self.fovea_input_counter = 0
        self.attentional_input_counter = 0
